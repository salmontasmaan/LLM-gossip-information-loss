"""check_sentences.py - check whether the sentences in the agents' final notes are correct.

For a random sample of final notes (default 6 per merge rule), every sentence is judged against the ORIGINAL
page by a fact-checking model, and also screened by code for numbers, months and acronyms that are not in the
source ("invented anchors").

Verdicts from the judge:
  SUPPORTED     the source states or clearly implies the sentence
  CONTRADICTED  the source says something different (another number, date or condition)
  UNSUPPORTED   the source does not say it (neither confirms nor contradicts)
A sentence counts as FULLY CORRECT only if it is SUPPORTED and contains no invented anchor.

This is descriptive: sentences within a note are not independent, so no significance tests are run here.
Needs the notes saved by run_experiment.py (<results>_notes.jsonl) - runs made before that was added have none.

Usage:  python check_sentences.py [--csv results/gossip_results.csv] [--per-rule 6] [--judge-model llama3.1:8b]"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

import config
from llm import Cache, _key                       # reuse the project's sqlite cache and key helper
from metrics import extract_anchors, split_sentences

LABELS = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
_LABEL_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED)\b", re.I)   # \b stops 'SUPPORTED' matching inside 'UNSUPPORTED'

SYSTEM = (
    "You are a strict fact-checker. You are given a SOURCE text and one STATEMENT. "
    "Decide whether the STATEMENT is supported by the SOURCE, using only the SOURCE and not your own knowledge. "
    "Answer with exactly one word on the first line: "
    "SUPPORTED (the source states or clearly implies it, including every number, date and condition), "
    "CONTRADICTED (the source says something different, for example another number, date or condition), or "
    "UNSUPPORTED (the source does not say this). "
    "On the second line give a reason in at most 20 words."
)


def parse_verdict(text: str) -> tuple[str, str]:
    """Turn the judge's reply into (label, reason). Unreadable replies become UNPARSED so they are counted, not hidden."""
    first, _, rest = text.strip().partition("\n")
    m = _LABEL_RE.search(first) or _LABEL_RE.search(text)    # prefer the first line, fall back to the whole reply
    label = m.group(1).upper() if m else "UNPARSED"
    return label, (rest.strip() or first.strip())[:200]


class OllamaJudge:
    """Fact-checker backed by a local Ollama model, with the same disk cache as the rest of the project."""

    def __init__(self, model: str):
        import ollama                              # imported here so the module can be tested without Ollama
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.model = model
        self.cache = Cache(config.CACHE_PATH)

    def judge(self, source: str, sentence: str) -> tuple[str, str]:
        user = f"SOURCE:\n{source}\n\nSTATEMENT:\n{sentence}"
        key = _key("judge", self.model, SYSTEM, user)          # includes the model, so different judges never mix
        reply = self.cache.get(key)
        if reply is None:
            resp = self.client.chat(model=self.model,
                                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                    options={"temperature": 0, "seed": 0, "num_predict": 80})   # deterministic, short
            reply = resp["message"]["content"].strip()
            self.cache.set(key, reply)
        return parse_verdict(reply)


def select_notes(notes: pd.DataFrame, per_rule: int, seed: int) -> pd.DataFrame:
    """Random sample of runs per rule; one note (agent 0) per run. Agents converge on similar notes anyway."""
    notes = notes.drop_duplicates(["doc_id", "rule", "seed", "agent"], keep="last")   # a crashed run may have been logged twice
    runs = notes[notes.agent == 0]
    rng = random.Random(seed)
    parts = []
    for rule in sorted(runs.rule.unique()):
        sub = runs[runs.rule == rule].reset_index(drop=True)
        take = rng.sample(range(len(sub)), min(per_rule, len(sub)))
        parts.append(sub.iloc[sorted(take)])
    return pd.concat(parts, ignore_index=True)


def check_notes(selected: pd.DataFrame, docs_by_id: dict, judge) -> pd.DataFrame:
    """One output row per sentence of every selected note."""
    rows = []
    for _, n in selected.iterrows():
        source = "\n".join(docs_by_id[n.doc_id]["sentences"])
        src_anchors = extract_anchors(source)
        for sentence in split_sentences(n.note):
            invented = sorted(extract_anchors(sentence) - src_anchors)    # numbers/months/acronyms not in the source
            label, reason = judge.judge(source, sentence)
            rows.append({"doc_id": n.doc_id, "rule": n.rule, "seed": n.seed, "agent": n.agent,
                         "sentence": sentence, "verdict": label, "invented_anchors": ";".join(invented),
                         "fully_correct": label == "SUPPORTED" and not invented, "reason": reason})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled shares per rule. Percentages are of all checked sentences for that rule."""
    out = []
    for rule in [r for r in config.RULES if r in set(df.rule)]:
        g = df[df.rule == rule]
        row = {"rule": rule, "notes": g.groupby(["doc_id", "seed"]).ngroups, "sentences": len(g)}
        for lab in LABELS + ["UNPARSED"]:
            row[lab.lower() + "_%"] = round(100 * (g.verdict == lab).mean(), 1)
        row["invented_anchor_%"] = round(100 * (g.invented_anchors != "").mean(), 1)
        row["fully_correct_%"] = round(100 * g.fully_correct.mean(), 1)
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.RESULTS_CSV), help="results CSV the notes belong to")
    ap.add_argument("--notes", default=None, help="notes file (default: derived from --csv)")
    ap.add_argument("--per-rule", type=int, default=config.JUDGE_NOTES_PER_RULE)
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--seed", type=int, default=0, help="seed for the random sample of notes")
    args = ap.parse_args()

    notes_path = config.notes_path_for(args.csv) if args.notes is None else Path(args.notes)
    if not notes_path.exists():
        raise SystemExit(f"No notes file at {notes_path}. Run run_experiment.py (the updated version saves the notes) first.")
    notes = pd.read_json(notes_path, lines=True)
    docs_by_id = {d["id"]: d for d in json.loads(config.DOCS_PATH.read_text(encoding="utf-8"))}

    selected = select_notes(notes, args.per_rule, args.seed)
    print(f"Checking {len(selected)} notes with judge {args.judge_model} "
          f"({selected.groupby('rule').size().to_dict()}) ...")
    df = check_notes(selected, docs_by_id, OllamaJudge(args.judge_model))

    out_csv = config.RESULTS_DIR / "sentence_check.csv"
    df.to_csv(out_csv, index=False)
    summary = summarise(df)
    summary.to_csv(config.RESULTS_DIR / "sentence_check_summary.csv", index=False)
    print("\nSentence-level correctness (share of checked sentences, per rule):\n")
    print(summary.to_string(index=False))
    print(f"\nPer-sentence verdicts saved to {out_csv}")
    print("Read the CONTRADICTED / UNSUPPORTED rows yourself: the judge is a model and can be wrong.")


if __name__ == "__main__":
    main()