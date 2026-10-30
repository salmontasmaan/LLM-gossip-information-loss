"""run_experiment.py - run every (document x rule x seed) gossip simulation and append results
to a CSV. Resumable: finished combinations are skipped on re-run.
Usage:  python run_experiment.py
        python run_experiment.py --rules lww llm_naive --out results/test.csv"""
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

import config
from gossip import run_gossip
from llm import OllamaLLM
from metrics import build_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS_CSV))
    ap.add_argument("--rules", nargs="+", default=config.RULES, help="subset of merge rules to run")
    args = ap.parse_args()

    docs = json.loads(config.DOCS_PATH.read_text(encoding="utf-8"))   # output of scrape.py
    llm = OllamaLLM()

    done = set()                              # (doc_id, rule, seed) combos already in the CSV
    try:
        prev = pd.read_csv(args.out)
        if "variant" in prev.columns:         # file written by the old multi-protocol version
            raise SystemExit(f"{args.out} has a 'variant' column from the earlier multi-protocol runs. "
                             "Use a fresh --out file (or delete it) so protocols are not mixed.")
        done = set(zip(prev.doc_id, prev.rule, prev.seed))
    except FileNotFoundError:
        pass

    jobs = [(d, rule, seed) for d in docs for rule in args.rules for seed in config.SEEDS
            if (d["id"], rule, seed) not in done]
    sources = {}                              # cache ground-truth objects per document
    for doc, rule, seed in tqdm(jobs, desc="gossip runs"):
        if doc["id"] not in sources:
            sources[doc["id"]] = build_source(doc["sentences"], llm)
        rows = run_gossip(doc["id"], doc["sentences"], sources[doc["id"]], rule, seed, llm)
        pd.DataFrame(rows).to_csv(args.out, mode="a", index=False,
                                  header=not os.path.exists(args.out))   # header only for a new file


if __name__ == "__main__":
    main()