"""llm.py - thin wrapper around Ollama (chat + embeddings) with a persistent cache."""
import hashlib                                # stable hashing for cache keys
import json
import sqlite3                                # stdlib key-value store, no extra install
import numpy as np

import config


class Cache:
    """Tiny sqlite-backed key/value store. Lets you stop and resume long experiments."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None    # None means "not cached"

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()


def _key(*parts) -> str:
    """Deterministic cache key from any JSON-serialisable parts."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _normalise(arr: np.ndarray) -> np.ndarray:
    """L2-normalise rows so that a dot product equals cosine similarity."""
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


class OllamaLLM:
    def __init__(self):
        import ollama                         # imported here so --mock works without the package
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.cache = Cache(config.CACHE_PATH)
        self.calls = 0                        # counts real (non-cached) generations

    def chat(self, system: str, user: str, seed: int = 0) -> str:
        key = _key("chat", config.CHAT_MODEL, system, user, seed, config.TEMPERATURE)
        hit = self.cache.get(key)
        if hit is not None:                   # identical request already answered earlier
            return hit
        resp = self.client.chat(
            model=config.CHAT_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": config.TEMPERATURE,   # sampling randomness
                     "seed": seed,                        # makes sampling reproducible
                     "num_predict": config.MAX_NEW_TOKENS},
        )
        text = resp["message"]["content"].strip()
        self.cache.set(key, text)
        self.calls += 1
        return text

    def embed(self, texts: list[str]) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]  # embedding models dislike empty strings
        out, missing = [None] * len(texts), []
        for i, t in enumerate(texts):         # first pass: use cached vectors
            hit = self.cache.get(_key("emb", config.EMBED_MODEL, t))
            if hit is None:
                missing.append(i)
            else:
                out[i] = hit
        if missing:                           # second pass: one batched call for the rest
            resp = self.client.embed(model=config.EMBED_MODEL, input=[texts[i] for i in missing])
            for i, vec in zip(missing, resp["embeddings"]):
                self.cache.set(_key("emb", config.EMBED_MODEL, texts[i]), vec)
                out[i] = vec
        return _normalise(np.array(out, dtype=np.float32))


def get_llm():
    return OllamaLLM()