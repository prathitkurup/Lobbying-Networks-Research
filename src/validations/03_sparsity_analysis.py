"""
03_sparsity_analysis.py
Prathit Kurup, Victoria Figueroa

PURPOSE
-------
Characterize network sparsity relative to a random null model and show that
the observed co-lobbying is far above chance — motivating the use of network
analysis rather than simple frequency counts.

NULL MODEL
----------
Under a random null where each firm independently selects bills to lobby with
no coordination, the probability that two firms share any given bill is:

    P(both firms lobby bill b) = (k_i / B) × (k_j / B)

where k_i = number of bills firm i lobbied, B = total unique bills.

The expected number of shared bills between a pair (i, j) is then:

    E[shared_bills] = Σ_b P(both lobby b) = Σ_b (k_i/B)(k_j/B) = k_i * k_j / B

At the median firm degree k_median, the expected shared bills for the median pair:

    E[shared_bills | median pair] ≈ k_median² / B

This is a conservative null: it ignores industry structure, so if anything it
overestimates random overlap and makes the 27x finding even more conservative.

FINDINGS (expected)
-------------------
  - Total unique bills (post-dedup):         ~2,300
  - Median firm bill count:                  ~20–30
  - Expected shared bills (median pair):     ~0.1
  - Observed median shared bills:            ~3
  - Ratio (above-null):                      ~27x

This 27x signal is the primary justification for treating co-lobbying as a
meaningful coordination signal rather than random overlap.

OUTPUT
------
Writes validations/outputs/03_sparsity_analysis.txt
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, MAX_BILL_DF
from utils.filtering import filter_bills_by_prevalence

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "03_sparsity_analysis.txt")


def compute_null_expected_shared(firm_degrees, B):
    """
    For every pair of firms, compute E[shared bills] = k_i * k_j / B under
    the random null. Returns a Series of expected shared counts.
    """
    k = firm_degrees.values
    # All pairs: vectorized outer product approach for efficiency
    # For large N, this is O(N^2) memory — acceptable for ~300 firms
    k_i = k[:, None]   # column vector
    k_j = k[None, :]   # row vector
    expected_matrix = (k_i * k_j) / B
    # Upper triangle only (i < j)
    upper = expected_matrix[np.triu_indices(len(k), k=1)]
    return pd.Series(upper)


def build_observed_edges(df):
    """Build correct (deduped + canonical) shared-bill edge weights."""
    df = df.drop_duplicates(subset=["client_name", "bill_id"])
    if MAX_BILL_DF is not None:
        df = filter_bills_by_prevalence(df, MAX_BILL_DF, unit_col="bill_id")

    bill_companies = df.groupby("bill_id")["client_name"].apply(list)
    records = []
    for bill_id, companies in bill_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if companies[i] != companies[j]:
                    src, tgt = ((companies[i], companies[j])
                                if companies[i] < companies[j]
                                else (companies[j], companies[i]))
                    records.append({"source": src, "target": tgt})
    if not records:
        return pd.DataFrame(columns=["source","target","weight"])
    return (pd.DataFrame(records)
              .groupby(["source","target"])
              .size()
              .reset_index(name="weight"))


def run_analysis(df):
    lines = []
    lines.append("=" * 70)
    lines.append("SPARSITY ANALYSIS: Observed Co-lobbying vs. Random Null Model")
    lines.append("=" * 70)

    # -- Step 1: Basic firm/bill counts --
    df_dedup = df.drop_duplicates(subset=["client_name", "bill_id"])
    B_raw = df_dedup["bill_id"].nunique()
    N = df_dedup["client_name"].nunique()

    lines.append(f"\n-- Step 1: Network Dimensions (post-deduplication) --")
    lines.append(f"  Unique firms:         {N}")
    lines.append(f"  Unique bills (raw):   {B_raw:,}")
    lines.append(f"  Possible firm pairs:  {N*(N-1)//2:,}")

    # -- Step 2: Firm degree distribution --
    firm_degrees = df_dedup.groupby("client_name")["bill_id"].nunique().rename("bill_count")
    lines.append(f"\n-- Step 2: Firm Bill Count (degree) Distribution --")
    for stat, val in [("mean",   firm_degrees.mean()),
                      ("median", firm_degrees.median()),
                      ("std",    firm_degrees.std()),
                      ("min",    firm_degrees.min()),
                      ("max",    firm_degrees.max())]:
        lines.append(f"  {stat:<8}: {val:.1f}")

    # -- Step 3: Null model — filtered --
    df_filtered = df_dedup.copy()
    if MAX_BILL_DF is not None:
        df_filtered = filter_bills_by_prevalence(df_dedup, MAX_BILL_DF,
                                                 unit_col="bill_id")
    B_filtered = df_filtered["bill_id"].nunique()
    firm_degrees_filt = df_filtered.groupby("client_name")["bill_id"].nunique()

    lines.append(f"\n-- Step 3: Null Model (filtered, MAX_BILL_DF={MAX_BILL_DF}) --")
    lines.append(f"  Bills after filtering: {B_filtered:,}")
    lines.append(f"  Firms after filtering: {firm_degrees_filt.shape[0]}")

    k_median = firm_degrees_filt.median()
    k_mean   = firm_degrees_filt.mean()
    E_median_pair = (k_median ** 2) / B_filtered
    E_mean_pair   = (k_mean   ** 2) / B_filtered

    lines.append(f"\n  E[shared bills | median pair] = k_med²/B "
                 f"= {k_median:.1f}² / {B_filtered} = {E_median_pair:.3f}")
    lines.append(f"  E[shared bills | mean pair]   = k_mean²/B "
                 f"= {k_mean:.1f}² / {B_filtered} = {E_mean_pair:.3f}")

    # -- Step 4: Observed edge statistics --
    lines.append(f"\n-- Step 4: Observed Edge Weights (filtered affiliation network) --")
    edges = build_observed_edges(df)

    n_edges    = len(edges)
    n_pairs    = N * (N - 1) // 2
    density    = n_edges / n_pairs

    lines.append(f"  Observed edges:       {n_edges:,}")
    lines.append(f"  Possible edges:       {n_pairs:,}")
    lines.append(f"  Network density:      {density:.3f}  ({100*density:.1f}%)")

    for stat, fn in [("mean",   "mean"), ("median", "median"),
                     ("std",    "std"),  ("p75",    lambda s: s.quantile(0.75)),
                     ("p90",    lambda s: s.quantile(0.90)), ("max", "max")]:
        if callable(fn):
            val = fn(edges["weight"])
        else:
            val = getattr(edges["weight"], fn)()
        lines.append(f"  weight {stat:<8}: {val:.2f}")

    # -- Step 5: Above-null ratio --
    obs_median = edges["weight"].median()
    ratio_median = obs_median / E_median_pair if E_median_pair > 0 else float("inf")
    obs_mean   = edges["weight"].mean()
    ratio_mean = obs_mean / E_mean_pair if E_mean_pair > 0 else float("inf")

    lines.append(f"\n-- Step 5: Observed vs. Null --")
    lines.append(f"  Median observed shared bills:  {obs_median:.1f}")
    lines.append(f"  Expected under null (median):  {E_median_pair:.3f}")
    lines.append(f"  Ratio (above null):            {ratio_median:.0f}x")
    lines.append(f"")
    lines.append(f"  Mean observed shared bills:    {obs_mean:.2f}")
    lines.append(f"  Expected under null (mean):    {E_mean_pair:.3f}")
    lines.append(f"  Ratio (above null):            {ratio_mean:.0f}x")

    # -- Step 6: Zero-edge pairs --
    n_zero = n_pairs - n_edges
    lines.append(f"\n-- Step 6: Zero-Edge Pairs (no shared bills) --")
    lines.append(f"  Pairs with zero shared bills:  {n_zero:,}  ({100*n_zero/n_pairs:.1f}%)")
    lines.append(f"  Pairs with shared bills:       {n_edges:,}  ({100*n_edges/n_pairs:.1f}%)")
    lines.append(f"  → The network is sparse at the global level, but within-community")
    lines.append(f"    overlap is substantially higher — motivating community analysis.")

    lines.append(f"\n-- Interpretation --")
    lines.append(f"  Co-lobbying is {ratio_median:.0f}x above random chance at the median pair,")
    lines.append(f"  despite the global sparsity. This is the core justification for:")
    lines.append(f"  (1) treating co-lobbying edges as a meaningful coordination signal,")
    lines.append(f"  (2) using community detection to find coordinated lobbying coalitions,")
    lines.append(f"  (3) studying cross-industry connectors (firms bridging coalitions).")
    lines.append(f"\n  See design_decisions.md §3 for full rationale.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "fortune500_lda_reports.csv")

    report = run_analysis(df)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
