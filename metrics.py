"""metrics.py - text utilities and the integrity metrics used to score an agent's note
against the ground-truth source document."""
import re                                     # regular expressions for anchor extraction
import numpy as np                            # vector maths for embeddings

# ----------------------------------------------------------- sentence splitting
# split after . ! ? when followed by a capital, digit, pound sign, quote or bracket
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z£\d\"'(])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")   # leading "- ", "* ", "1. " markers


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Each line is treated as at least one sentence,
    so bullet lists and table rows survive."""
    out = []
    for line in text.splitlines():            # newline is a hard sentence boundary
        line = _BULLET.sub("", line).strip()  # drop bullet/number markers
        if not line:                          # skip blank lines
            continue
        out.extend(p.strip() for p in _SENT_SPLIT.split(line) if p.strip())
    return out


# ------------------------------------------------------------------- anchors
# "Anchors" are cheap-to-verify atoms of information: numbers, months, acronyms.
# If an anchor in the source is missing from a note, that fact was lost.
_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")        # 12,570  /  20  /  2.5
_MONTH = re.compile(r"\b(January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)\b")
_ACRO = re.compile(r"\b[A-Z]{2,}\b")                  # HMRC, VAT, NI, UK ...


def extract_anchors(text: str) -> set[str]:
    """Return the set of normalised anchors found in text."""
    anchors = set()
    for m in _NUM.findall(text):              # numbers, with thousands commas removed
        anchors.add(m.replace(",", ""))       # so "12,570" == "12570"
    for m in _MONTH.findall(text):            # month names, lower-cased
        anchors.add(m.lower())
    for m in _ACRO.findall(text):             # acronyms kept as-is
        anchors.add(m)
    return anchors


# -------------------------------------------------------------- source object
def build_source(sentences: list[str], llm) -> dict:
    """Pre-compute everything about the ground truth once per document."""
    text = "\n".join(sentences)
    return {
        "text": text,
        "sentences": sentences,
        "anchors": extract_anchors(text),
        "sent_emb": llm.embed(sentences),     # (n_sentences, dim), L2-normalised
        "doc_emb": llm.embed([text])[0],      # (dim,), L2-normalised
    }


# ------------------------------------------------------------- state scoring
def evaluate_state(state: str, source: dict, llm, threshold: float) -> dict:
    """Score one agent's note against the source. All metrics are in [0, 1]."""
    anchors = extract_anchors(state)
    src = source["anchors"]
    hit = anchors & src                       # anchors the note kept correctly
    recall = len(hit) / len(src)              # fraction of source anchors retained
    precision = len(hit) / len(anchors) if anchors else 1.0  # 1 - precision = fabricated anchors

    sents = split_sentences(state)
    if sents:
        emb = llm.embed(sents)                # embed the note's sentences
        sims = source["sent_emb"] @ emb.T     # cosine matrix: source sentences x note sentences
        best = sims.max(axis=1)               # best-matching note sentence for each source sentence
        coverage = float((best >= threshold).mean())   # share of source sentences "still present"
        mean_best = float(best.mean())        # threshold-free version of the same idea
        doc_cos = float(source["doc_emb"] @ llm.embed([state])[0])  # whole-document similarity
    else:                                     # empty note: everything lost
        coverage = mean_best = doc_cos = 0.0

    return {
        "anchor_recall": recall,
        "anchor_precision": precision,
        "coverage": coverage,
        "mean_best_sim": mean_best,
        "doc_cosine": doc_cos,
        "n_words": float(len(state.split())),
    }


def consensus(states: list[str], llm) -> float:
    """Mean pairwise cosine between agents' notes (1.0 = everyone holds the same thing)."""
    emb = llm.embed(states)
    sims = emb @ emb.T
    iu = np.triu_indices(len(states), k=1)    # upper triangle = each unordered pair once
    return float(sims[iu].mean())