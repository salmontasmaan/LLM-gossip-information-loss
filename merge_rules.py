"""merge_rules.py - the reconciliation rules under comparison.

Every rule is a pair (init_fn, merge_fn):
    init_fn(shard_sentences, ctx) -> initial note (str) for one agent
    merge_fn(note_a, note_b, ctx) -> reconciled note (str) that BOTH peers adopt
`ctx` carries the llm, seed, word budget and running counters."""
import re

import config
from metrics import extract_anchors, split_sentences

# ------------------------------------------------------------------- prompts
SYS_NAIVE = ("You merge two notes into one note. Keep it under {max_words} words. "
             "Output only the merged note, no preamble.")

SYS_ANCHOR = (
    "You are a careful note-merger for UK government guidance. Rules: "
    "1) Keep every number, amount, percentage, date, deadline and named scheme or organisation "
    "exactly as written. "
    "2) Do not add any information that is not in the notes. "
    "3) If the notes contradict each other, keep both versions and mark the clash with 'CONFLICT'. "
    "4) Remove only exact duplicates; do not summarise. "
    "5) Keep it under {max_words} words. Output only the merged note as plain sentences, no preamble."
)

SYS_CLAIMS = ("Break the text into atomic factual claims, one claim per line. Each claim must be "
              "self-contained and keep every number, date and name exactly as written. "
              "Output only the lines, no numbering, no preamble.")


def _user_prompt(a: str, b: str, max_words: int) -> str:
    return f"NOTE A:\n{a}\n\nNOTE B:\n{b}\n\nMerged note (max {max_words} words):"


_PREAMBLE = re.compile(r"^(here('s| is)|sure|certainly|merged note)[^\n]*:\s*$", re.I)


def _clean(text: str) -> str:
    """Remove chatty first lines such as 'Here is the merged note:' that small models add."""
    lines = text.strip().splitlines()
    if lines and _PREAMBLE.match(lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _norm(s: str) -> str:
    """Normalise a sentence for exact-duplicate detection."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9£%. ]", "", s.lower())).strip()


def _llm_merge(system_tpl: str, a: str, b: str, ctx: dict) -> str:
    """Shared LLM call. Falls back to plain concatenation if the model returns nothing."""
    out = _clean(ctx["llm"].chat(system_tpl.format(max_words=ctx["max_words"]),
                                 _user_prompt(a, b, ctx["max_words"]), seed=ctx["seed"]))
    if not out:                               # empty output = failure; log it, don't hide it
        ctx["fallbacks"] += 1
        return a + "\n" + b
    return out


# ------------------------------------------------------------- default init
def init_raw(shard: list[str], ctx: dict) -> str:
    """Agent starts holding its slice of the source text verbatim."""
    return "\n".join(shard)


def init_claims(shard: list[str], ctx: dict) -> str:
    """Agent converts its slice to atomic claims ONCE (the only LLM touch for this rule)."""
    out = _clean(ctx["llm"].chat(SYS_CLAIMS, "\n".join(shard), seed=ctx["seed"]))
    claims = split_sentences(out)
    return "\n".join(claims) if claims else "\n".join(shard)   # fallback to raw if extraction failed


# -------------------------------------------------------------------- rules
def merge_lww(a: str, b: str, ctx: dict) -> str:
    """Last-writer-wins: the peer's note overwrites ours. Classic distributed-systems baseline;
    here it throws away one agent's knowledge every exchange (lower bound)."""
    return b


def merge_llm_naive(a: str, b: str, ctx: dict) -> str:
    """Ask the LLM to merge, with no special care. Represents 'just let the model handle it'."""
    return _llm_merge(SYS_NAIVE, a, b, ctx)


def merge_llm_anchor(a: str, b: str, ctx: dict) -> str:
    """LLM merge with an explicit preservation/conflict-marking prompt (prompt-level fix)."""
    return _llm_merge(SYS_ANCHOR, a, b, ctx)


def merge_llm_repair(a: str, b: str, ctx: dict) -> str:
    """LLM merge + deterministic verification. Any anchor present in an input but missing from
    the output is repaired by re-appending the original sentence(s) that carried it."""
    merged = merge_llm_anchor(a, b, ctx)
    missing = (extract_anchors(a) | extract_anchors(b)) - extract_anchors(merged)
    if not missing:                           # nothing lost, accept the LLM's output
        return merged
    appended = []
    for sent in split_sentences(a) + split_sentences(b):
        covers = extract_anchors(sent) & missing
        if covers:                            # this sentence restores at least one lost anchor
            appended.append(sent)
            missing -= covers                 # greedy set cover: don't re-add redundant sentences
    ctx["repairs"] += len(appended)           # counter lets us report how often repair was needed
    return merged + "\n" + "\n".join(appended)


def merge_claim_union(a: str, b: str, ctx: dict) -> str:
    """CRDT-style grow-only set: notes are lists of atomic claims; merge = set union with exact
    de-duplication. No LLM rewriting at merge time, so loss can only come from the initial extraction."""
    seen, out = set(), []
    for claim in split_sentences(a) + split_sentences(b):
        k = _norm(claim)
        if k and k not in seen:               # keep first occurrence of each claim
            seen.add(k)
            out.append(claim)
    return "\n".join(out)


# registry: rule name -> (init, merge)
RULES = {
    "lww": (init_raw, merge_lww),
    "llm_naive": (init_raw, merge_llm_naive),
    "llm_anchor": (init_raw, merge_llm_anchor),
    "llm_repair": (init_raw, merge_llm_repair),
    "claim_union": (init_claims, merge_claim_union),
}