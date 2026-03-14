"""
02_inflation_diagnosis.py
Prathit Kurup, Victoria Figueroa

PURPOSE
-------
Demonstrate and quantify the cartesian product inflation bug in the original
network construction code, and verify that the aggregation fix produces correct
edge weights.

THE BUG
-------
The original company_bill_edges() called:
    df.groupby("bill_id")["client_name"].apply(list)

on the raw (un-aggregated) DataFrame. When firm A has R_A rows for bill b and
firm B has R_B rows, the list for bill b contains R_A copies of A and R_B copies
of B. The i < j inner loop then produces R_A × R_B pair records instead of 1,
inflating the shared-bill count for each pair by a factor equal to the product
of their per-bill row counts.

Example:
  Boeing lobbied hr2923-116 in 5 different reports → 5 rows
  Lockheed lobbied hr2923-116 in 4 different reports → 4 rows
  Inflated count for Boeing-Lockheed on that one bill: 5 × 4 = 20 records
  True count: 1

At the median, this produced an inflation factor of ~6x
(median shared bills: 19 inflated vs. 3 corrected).

THE FIX
-------
For the affiliation network:
    df = df.drop_duplicates(subset=["client_name", "bill_id"])
    → reduces each (firm, bill) pair to a single presence/absence row

For the BC similarity network:
    df = df.groupby(["client_name","bill_id"], as_index=False)["amount"].sum()
    → collapses to true total allocated spend per (firm, bill)

ALSO FIXED: canonical pair ordering
Without canonicalization, (A, B) and (B, A) from different bill/registrant
list orderings create separate edge records that don't merge correctly in the
groupby. Fix: src, tgt = (a,b) if a < b else (b, a).

OUTPUT
------
Writes validations/outputs/02_inflation_diagnosis.txt
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "02_inflation_diagnosis.txt")


def build_edges_buggy(df):
    """Original (buggy) implementation — no deduplication, no canonical ordering."""
    bill_companies = df.groupby("bill_id")["client_name"].apply(list)
    records = []
    for bill_id, companies in bill_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if companies[i] != companies[j]:
                    records.append({"source": companies[i], "target": companies[j]})
    return (pd.DataFrame(records)
              .groupby(["source", "target"])
              .size()
              .reset_index(name="weight"))


def build_edges_fixed(df):
    """Fixed implementation — deduplication + canonical ordering."""
    df = df.drop_duplicates(subset=["client_name", "bill_id"])
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
    return (pd.DataFrame(records)
              .groupby(["source", "target"])
              .size()
              .reset_index(name="weight"))


def run_diagnosis(df):
    lines = []
    lines.append("=" * 70)
    lines.append("INFLATION DIAGNOSIS: Cartesian Product Bug in Edge Construction")
    lines.append("=" * 70)

    lines.append(f"\n-- Step 1: Quantify Raw Duplication --")
    pair_counts = df.groupby(["client_name", "bill_id"]).size()
    lines.append(f"  Rows per (firm, bill) pair:")
    lines.append(f"    Mean:   {pair_counts.mean():.2f}")
    lines.append(f"    Median: {pair_counts.median():.1f}")
    lines.append(f"    Max:    {pair_counts.max()}")
    lines.append(f"  Under the bug, for a pair of firms with R_A and R_B rows on a")
    lines.append(f"  shared bill, the loop produces R_A × R_B records instead of 1.")
    lines.append(f"  Expected median inflation factor ≈ {pair_counts.median()**2:.1f}x "
                 f"(median R^2).")

    lines.append(f"\n-- Step 2: Buggy vs. Fixed Edge Weights --")
    lines.append(f"  Building edges with buggy code (no dedup)...")
    buggy = build_edges_buggy(df)
    lines.append(f"  Building edges with fixed code (dedup + canonical)...")
    fixed = build_edges_fixed(df)

    lines.append(f"\n  BUGGY edges:  {len(buggy):,} pairs")
    lines.append(f"    weight stats:  mean={buggy['weight'].mean():.1f}  "
                 f"median={buggy['weight'].median():.1f}  "
                 f"max={buggy['weight'].max()}")

    lines.append(f"\n  FIXED edges:  {len(fixed):,} pairs")
    lines.append(f"    weight stats:  mean={fixed['weight'].mean():.1f}  "
                 f"median={fixed['weight'].median():.1f}  "
                 f"max={fixed['weight'].max()}")

    inflation = buggy['weight'].median() / fixed['weight'].median()
    lines.append(f"\n  Inflation factor at median: {inflation:.1f}x  "
                 f"({buggy['weight'].median():.0f} buggy / {fixed['weight'].median():.0f} fixed)")

    lines.append(f"\n-- Step 3: Example Pair --")
    # Find the pair with largest difference
    # Merge on canonical ordering
    buggy_canon = buggy.copy()
    mask = buggy_canon["source"] > buggy_canon["target"]
    buggy_canon.loc[mask, ["source","target"]] = \
        buggy_canon.loc[mask, ["target","source"]].values
    buggy_grp = (buggy_canon.groupby(["source","target"])["weight"].sum()
                             .reset_index())

    merged = fixed.merge(buggy_grp, on=["source","target"],
                         suffixes=("_fixed","_buggy"))
    merged["ratio"] = merged["weight_buggy"] / merged["weight_fixed"]
    worst = merged.nlargest(1, "ratio").iloc[0]
    lines.append(f"  Worst inflated pair:")
    lines.append(f"    {worst['source']}  ↔  {worst['target']}")
    lines.append(f"    Fixed weight:  {int(worst['weight_fixed'])} shared bills")
    lines.append(f"    Buggy weight:  {int(worst['weight_buggy'])} (≈{worst['ratio']:.0f}x inflated)")

    lines.append(f"\n-- Step 4: Canonical Pair Ordering --")
    # Check for non-canonical pairs in buggy output
    non_canonical = buggy[buggy["source"] > buggy["target"]]
    lines.append(f"  Non-canonical pairs in buggy output: {len(non_canonical):,}")
    lines.append(f"  (source > target — these are stored separately from their")
    lines.append(f"   canonical counterpart, causing double-counting.)")
    lines.append(f"  Fixed output: 0 non-canonical pairs (guaranteed by src<tgt constraint).")

    lines.append(f"\n-- Summary of Fixes --")
    lines.append(f"  1. drop_duplicates(subset=['client_name','bill_id'])")
    lines.append(f"     Reduces {len(df):,} raw rows to "
                 f"{df.drop_duplicates(['client_name','bill_id']).shape[0]:,} unique (firm, bill) pairs.")
    lines.append(f"  2. src, tgt = (a,b) if a<b else (b,a)")
    lines.append(f"     Canonical ordering ensures (A,B) and (B,A) always merge.")
    lines.append(f"  Result: median shared-bill count corrected from "
                 f"{buggy['weight'].median():.0f} → {fixed['weight'].median():.0f}.")
    lines.append(f"\n  See design_decisions.md §1-2 for full rationale.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "fortune500_lda_reports.csv")

    report = run_diagnosis(df)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
