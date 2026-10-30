"""analyze.py - descriptive final-round summary table for the results CSV.
(The hypothesis tests are in hypothesis_tests.py.)
Usage:  python analyze.py [--csv path]"""
import argparse

import numpy as np
import pandas as pd

import config

# loss is the primary outcome; the rest are secondary metrics reported without further testing
METRICS = ["loss", "hallucination", "anchor_recall", "coverage", "mean_best_sim", "consensus"]


def bootstrap_median_ci(x, n=5000, seed=0):
    """95% percentile-bootstrap confidence interval for the median."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.RESULTS_CSV))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if "variant" in df.columns:               # file written by the old multi-protocol version
        raise SystemExit("This CSV has a 'variant' column from the earlier multi-protocol runs. "
                         "Re-run into a fresh results file so protocols are not mixed.")
    df["hallucination"] = 1 - df["anchor_precision"]       # share of anchors the note invented
    df["loss"] = 1 - df["pool_anchor_recall"]              # primary outcome: details lost from the whole network

    last = df[df["round"] == df["round"].max()]
    per_doc = last.groupby(["doc_id", "rule"], as_index=False)[METRICS].mean()   # average seeds -> one value per page
    rows = []
    for rule in [r for r in config.RULES if r in set(per_doc.rule)]:
        g = per_doc[per_doc.rule == rule]
        row = {"rule": rule, "n_docs": len(g)}
        for m in METRICS:
            lo, hi = bootstrap_median_ci(g[m])
            row[m] = f"{g[m].median():.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "summary_final_round.csv", index=False)
    print("\nFinal-round summary (median [95% bootstrap CI], one value per page):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved {config.RESULTS_DIR / 'summary_final_round.csv'}")


if __name__ == "__main__":
    main()