"""
BC similarity diagnostic utilities.

These functions are used to inspect and validate the breadth × depth
Bray-Curtis edge construction. They are separated from the main network
script to keep it concise; import and call them individually as needed.

  duplicate_check(df_raw)
      Quantify (client_name, bill_id) duplication before aggregation.
      Confirms that the groupby-sum aggregation fix is necessary.

  diagnostic_summary(edges)
      Distribution of depth, breadth, and final weight.

  top_edges_inspection(edges, k=20)
      Top-k edges with driver classification (depth-driven, balanced, etc.)

  robustness_check(edges_fn, df, lambdas)
      Spearman ρ of final weights at alternate λ values vs. the auto-calibrated
      baseline. Stable rankings support λ-insensitivity claims.
"""

import numpy as np
import pandas as pd


def duplicate_check(df_raw):
    """
    Report how many (client_name, bill_id) pairs have multiple rows before
    aggregation. Run once to confirm aggregation is necessary and to quantify
    the inflation factor (see validations/02_inflation_diagnosis.py for the
    full side-by-side comparison).
    """
    counts = df_raw.groupby(["client_name", "bill_id"]).size()
    n_multi = (counts > 1).sum()
    n_total = len(counts)
    print(f"\n-- Pre-aggregation Duplicate Check --")
    print(f"  Unique (firm, bill) pairs:       {n_total:,}")
    print(f"  Pairs with multiple rows:        {n_multi:,}  ({100*n_multi/n_total:.1f}%)")
    print(f"  Max rows for a single pair:      {counts.max()}")
    print(f"  Mean rows per pair (multi only): "
          f"{counts[counts > 1].mean():.2f}")


def diagnostic_summary(edges):
    """Print distribution of the three components driving edge weight."""
    print("\n-- Breadth × Depth Diagnostic --")
    for col, label in [("shared_bills", "Shared bills"),
                       ("depth",        "Depth (mean BC)"),
                       ("breadth",      "Breadth (1-exp)"),
                       ("weight",       "Final weight")]:
        s = edges[col]
        print(f"  {label:<22}  mean={s.mean():.3f}  std={s.std():.3f}  "
              f"p25={s.quantile(.25):.3f}  p50={s.quantile(.50):.3f}  "
              f"p75={s.quantile(.75):.3f}")


def top_edges_inspection(edges, k=20):
    """
    Print the top-k edges by weight with depth and breadth broken out.

    Driver classification:
      saturated       breadth ≥ 0.95  (weight ≈ depth; breadth fully rewarded)
      breadth-boosted breadth ≥ 0.50 AND depth < 0.40
                      (breadth lifting a weak-depth pair — check for validity)
      depth-driven    depth > breadth by more than 0.15
      balanced        depth and breadth within 0.15 of each other
      weak-overlap    breadth < 0.50  (limited co-lobbying breadth)
    """
    top = edges.nlargest(k, "weight").copy()

    def classify(r):
        b, d = r["breadth"], r["depth"]
        if b >= 0.95:
            return "saturated"
        if b >= 0.50 and d < 0.40:
            return "breadth-boosted"
        if b < 0.50:
            return "weak-overlap"
        if abs(b - d) <= 0.15:
            return "balanced"
        return "depth-driven" if d > b else "breadth-boosted"

    top["driver"] = top.apply(classify, axis=1)

    print(f"\n-- Top {k} Edges by Weight --")
    print(f"  {'Source':<32} {'Target':<32} {'Weight':>7} "
          f"{'Bills':>6} {'Depth':>7} {'Breadth':>8} {'Driver'}")
    print(f"  {'-'*32} {'-'*32} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*15}")
    for _, r in top.iterrows():
        print(f"  {r['source']:<32} {r['target']:<32} {r['weight']:>7.3f} "
              f"{int(r['shared_bills']):>6} {r['depth']:>7.3f} "
              f"{r['breadth']:>8.3f}  {r['driver']}")

    counts = top["driver"].value_counts()
    print(f"\n  Driver breakdown in top {k}: "
          + "  |  ".join(f"{d}: {n}" for d, n in counts.items()))
    if "breadth-boosted" in counts:
        print(f"\n  *** {counts['breadth-boosted']} breadth-boosted edge(s) — "
              f"review for validity ***")


def robustness_check(edges_fn, df, lambdas):
    """
    Spearman rank correlation of final weights at alternate λ values vs. the
    auto-calibrated baseline. Stable rankings (ρ ≥ 0.95) support the claim
    that results are not sensitive to λ calibration.
    """
    from scipy.stats import spearmanr

    print("\n-- λ Robustness Check --")
    baseline = edges_fn(df, lam=None).set_index(["source", "target"])["weight"]

    for lam in lambdas:
        alt = edges_fn(df, lam=lam).set_index(["source", "target"])["weight"]
        shared_idx = baseline.index.intersection(alt.index)
        rho, _ = spearmanr(baseline.loc[shared_idx], alt.loc[shared_idx])
        print(f"  λ = {lam:.5f}   Spearman ρ vs baseline = {rho:.4f}")
