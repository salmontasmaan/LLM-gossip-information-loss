"""gossip.py - push-pull gossip simulation over a fully connected network of LLM agents.
This is the single protocol used in the study; only the merge rule changes between conditions."""
import random
import numpy as np

import config
from merge_rules import RULES
from metrics import evaluate_state, extract_anchors, consensus


def make_shards(sentences: list[str], n: int) -> list[list[str]]:
    """Split the document into n contiguous, near-equal slices (one per agent)."""
    base, extra = divmod(len(sentences), n)   # base size, plus 1 extra sentence for the first `extra` shards
    shards, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        shards.append(sentences[start:start + size])
        start += size
    return shards


def run_gossip(doc_id: str, sentences: list[str], source: dict, rule: str, seed: int, llm):
    """Run one full simulation. Returns (rows, final_notes): one metrics row per round (round 0 = before gossip)
    and each agent's final note (needed by check_sentences.py)."""
    init_fn, merge_fn = RULES[rule]
    rng = random.Random(seed)                 # controls peer selection -> reproducible topology
    n = config.N_AGENTS
    ctx = {"llm": llm, "seed": seed, "max_words": config.MAX_NOTE_WORDS,
           "fallbacks": 0, "repairs": 0, "merges": 0}   # counters are filled in by the merge rules / the loop below
    states = [init_fn(shard, ctx) for shard in make_shards(sentences, n)]  # each agent knows 1/n of the doc
    rows = []

    def snapshot(r: int):
        """Score every agent, average, and also compute network-level metrics."""
        per_agent = [evaluate_state(s, source, llm, config.COVERAGE_THRESHOLD) for s in states]
        row = {k: float(np.mean([m[k] for m in per_agent])) for k in per_agent[0]}  # mean over agents
        pooled = set().union(*(extract_anchors(s) for s in states))   # anchors alive anywhere in network
        row["pool_anchor_recall"] = len(pooled & source["anchors"]) / len(source["anchors"])
        row["consensus"] = consensus(states, llm)
        row.update(doc_id=doc_id, rule=rule, seed=seed, round=r,
                   fallbacks=ctx["fallbacks"], repairs=ctx["repairs"], merges=ctx["merges"])
        rows.append(row)

    snapshot(0)                               # baseline: what each agent knows before any gossip
    for r in range(1, config.N_ROUNDS + 1):
        order = list(range(n))
        rng.shuffle(order)                    # random initiator order each round
        for i in order:
            j = rng.choice([k for k in range(n) if k != i])   # pick one random peer (not self)
            merged = merge_fn(states[i], states[j], ctx)      # reconcile the two notes ONCE...
            states[i] = states[j] = merged    # ...and both adopt it (push-pull exchange)
            ctx["merges"] += 1                # one merge call per exchange (cost accounting)
        snapshot(r)
    return rows, list(states)