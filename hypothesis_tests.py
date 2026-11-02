"""hypothesis_tests.py - Wilcoxon signed-rank tests for RQ1 and RQ2.

Loss L = 1 - pool_anchor_recall at the final round: the share of the source's verifiable details
(numbers, months, acronyms) that NO agent holds any more. Unit of analysis = one GOV.UK page
(seeds are averaged first), so each test sees one independent value per page.

The Wilcoxon signed-rank test does not need normality. It tests the median (more precisely the centre of
a roughly symmetric distribution) and drops pages that tie exactly with the tested value.

RQ1, one-sample, one-sided test on EVERY rule:   H0: median L <= EPSILON   vs  H1: median L > EPSILON
RQ2, paired, one-sided test, rule vs llm_naive:  H0: median(L_naive - L_rule) <= 0  vs  H1: > 0
Each family is Holm-corrected across the rules in it.

Usage:  python hypothesis_tests.py [--csv path] [--round N]"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config

NAIVE = "llm_naive"                                        # the unguarded LLM merge = reference condition
CANDIDATES = ["llm_anchor", "llm_repair", "claim_union"]   # rules claimed to reduce loss (lww is a floor, not a candidate)


def holm(pvals: dict) -> dict:
    """Holm step-down adjustment. NaN p-values (undefined tests) are left out of the family."""
    valid = {k: p for k, p in pvals.items() if not np.isnan(p)}
    adj, running = {k: np.nan for k in pvals}, 0.0
    m = len(valid)
    for i, (k, p) in enumerate(sorted(valid.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))      # enforce monotonic adjusted p-values
        adj[k] = running
    return adj


def bootstrap_median_ci(x, n_boot=5000, seed=0):
    """95% percentile-bootstrap confidence interval for the median."""
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])


def signed_rank(x, mu: float, alternative: str) -> dict:
    """One-sample Wilcoxon signed-rank test of the median of x against mu.
    For a paired comparison pass the per-page differences and mu=0.
    alternative: 'greater' (median > mu) or 'less' (median < mu)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    diff = x - mu
    n_used = int(np.count_nonzero(diff))                   # pages that differ from mu (exact ties are dropped)
    out = {"n": n, "n_used": n_used, "median": float(np.median(x)) if n else np.nan,
           "ci_lo": np.nan, "ci_hi": np.nan, "W": np.nan, "p": np.nan,
           "min_p": 0.5 ** n_used if n_used else np.nan,   # smallest one-sided p this many pages can ever give
           "note": ""}
    if n >= 2:
        out["ci_lo"], out["ci_hi"] = bootstrap_median_ci(x)
    if n_used == 0:
        out["note"] = "every page equals the tested value; test undefined"
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # SciPy warns about small samples / ties; handled in notes
        res = wilcoxon(diff, alternative=alternative, zero_method="wilcox", method="auto")
    out["W"], out["p"] = float(res.statistic), float(res.pvalue)
    if n_used < n:
        out["note"] = f"{n - n_used} page(s) tied with the tested value and were dropped"
    return out


def verdict(p: float, reject_text: str, keep_text: str) -> str:
    if np.isnan(p):
        return "n/a"
    return reject_text if p < config.ALPHA else keep_text


def warn_if_underpowered(results: dict, label: str):
    """Holm needs the smallest p <= alpha/m for ANY rejection; a signed-rank test with n pages cannot go below 0.5**n."""
    m = sum(not np.isnan(r["p"]) for r in results.values())
    best = min((r["min_p"] for r in results.values() if not np.isnan(r["min_p"])), default=np.nan)
    if m and not np.isnan(best) and best > config.ALPHA / m:
        print(f"  ! {label}: with this many pages the smallest possible p ({best:.4f}) is above alpha/m "
              f"({config.ALPHA / m:.4f}); no rule can be rejected after Holm correction. Add more pages.")


def load_loss(csv_path: str, rnd) -> tuple[pd.DataFrame, int]:
    """Return a pages x rules table of final-round loss, averaged over seeds."""
    df = pd.read_csv(csv_path)
    if "variant" in df.columns:                            # file written by the old multi-protocol version
        raise SystemExit("This CSV has a 'variant' column from the earlier multi-protocol runs. "
                         "Re-run into a fresh results file so protocols are not mixed.")
    rnd = int(df["round"].max()) if rnd is None else int(rnd)
    df = df[df["round"] == rnd].copy()
    df["loss"] = 1.0 - df["pool_anchor_recall"]
    return df.groupby(["doc_id", "rule"])["loss"].mean().unstack("rule"), rnd   # average seeds -> one value per page


def fmt_ci(r: dict) -> str:
    return "n/a" if np.isnan(r["ci_lo"]) else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def report(rows: list, test: str, results: dict, adj: dict, label_median: str, reject_text: str, eps=np.nan):
    """Print one family of tests and append its rows to the CSV list."""
    for rule, r in results.items():
        print(f"  {rule:12s} {label_median}={r['median']:+.3f} 95%CI {fmt_ci(r)} n={r['n']} (used {r['n_used']}) "
              f"W={r['W']:.1f} p={r['p']:.4g} p_holm={adj[rule]:.4g} -> "
              f"{verdict(adj[rule], reject_text, 'fail to reject H0')} {r['note']}")
        rows.append({"test": test, "rule": rule, "eps": eps, **r, "p_holm": adj[rule],
                     "decision": verdict(adj[rule], "reject H0", "fail to reject H0")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.RESULTS_CSV))
    ap.add_argument("--round", type=int, default=None, help="round to test (default: last)")
    args = ap.parse_args()

    loss, rnd = load_loss(args.csv, args.round)
    eps, alpha = config.EPSILON, config.ALPHA
    if NAIVE not in loss:
        raise SystemExit(f"{NAIVE} not in results (needed as the reference for RQ2)")
    all_rules = [r for r in config.RULES if r in loss.columns]
    rows = []
    print(f"\nRound {rnd} | {len(loss)} pages | tolerable loss eps={eps} | alpha={alpha}")
    print("Median loss per rule (reference):  " + ", ".join(f"{r}={loss[r].median():.3f}" for r in all_rules))

    # ------------------------------------------------------------------ RQ1
    print("\n=== RQ1: is loss above the tolerable level, for each rule? ===")
    print(f"H0: median loss <= {eps}   |   H1: median loss > {eps}   (one-sample, one-sided Wilcoxon signed-rank, "
          f"Holm-adjusted across {len(all_rules)} rules)")
    res = {rule: signed_rank(loss[rule].dropna(), eps, "greater") for rule in all_rules}
    adj = holm({k: v["p"] for k, v in res.items()})
    report(rows, "RQ1", res, adj, "median", "reject H0 (loss exceeds tolerance)", eps)
    warn_if_underpowered(res, "RQ1")

    # ----------------------------------------------------------------- RQ2
    rules = [r for r in CANDIDATES if r in loss.columns]
    print("\n=== RQ2: does each rule reduce loss relative to the unguarded LLM merge? ===")
    print("H0: median(L_naive - L_rule) <= 0   |   H1: median(L_naive - L_rule) > 0   "
          "(paired, one-sided Wilcoxon signed-rank, Holm-adjusted)")
    res = {rule: signed_rank((loss[NAIVE] - loss[rule]).dropna(), 0.0, "greater") for rule in rules}   # pair by page
    adj = holm({k: v["p"] for k, v in res.items()})
    report(rows, "RQ2", res, adj, "median reduction", "reject H0 (reduces loss)")
    warn_if_underpowered(res, "RQ2")

    out = config.RESULTS_DIR / "hypothesis_tests.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()