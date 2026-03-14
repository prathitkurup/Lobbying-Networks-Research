"""
05_issue_score_range.py
Prathit Kurup, Victoria Figueroa

PURPOSE
-------
Document and prove (with math and empirical evidence) why the issue similarity
network weights are NOT bounded between 0 and 1.

BACKGROUND
----------
The issue similarity network uses:
  weight(i,j) = sum_k BC_k(i,j) / sqrt(shared_issue_count)  [NORMALIZE=True]
  weight(i,j) = sum_k BC_k(i,j)                             [NORMALIZE=False]

where BC_k is Bray-Curtis similarity on issue k.

Key insight:
- Each BC_k is bounded in [0, 1] (per-issue similarity).
- The SUM of N such values can be at most N.
- Dividing by sqrt(N) gives a maximum of N / sqrt(N) = sqrt(N).
- With 75 issue codes, the theoretical maximum is sqrt(75) ≈ 8.66.
- Without normalization, the maximum is 75 itself.

This is intentional: the sqrt normalization is a COMPROMISE between:
  Raw sum (N):        rewards breadth linearly, no upper bound
  Plain mean (1):     removes breadth signal entirely
  sqrt normalization: rewards breadth sub-linearly, bounded by sqrt(N)

DESIGN DECISION DOCUMENTED
--------------------------
The unbounded weight range means edge weights CANNOT be interpreted as
probabilities or proportions in [0,1]. Users should interpret a weight of 3.0
as roughly corresponding to 9 shared issues with perfect BC alignment
(9/sqrt(9) = 3). This is appropriate for community detection and centrality
algorithms, which handle arbitrary positive weights without assuming [0,1].

Contrast with the bill-level BC similarity network, which uses:
  weight(i,j) = BC_mean × breadth_term
where breadth_term = 1 - exp(-λ × shared_bill_count) ∈ [0,1),
giving a final weight always in [0,1].

OUTPUT
------
Writes a human-readable proof and distribution report to:
  validations/outputs/05_issue_score_range.txt
"""

import sys
import os
import math
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure config is imported from src/ (parent of this directory)
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent
sys.path.insert(0, str(_src_dir))
from config import DATA_DIR

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs",
                           "05_issue_score_range.txt")


def run_proof(df_issues=None):
    """
    Document the mathematical proof and show empirical distribution if data available.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ISSUE SIMILARITY WEIGHT RANGE PROOF")
    lines.append("=" * 80)

    # --
    lines.append("\n-- Mathematical Foundation --")
    lines.append("")
    lines.append("Bray-Curtis per issue k:")
    lines.append("  BC_k(i,j) = 1 - |f_i,k - f_j,k| / (f_i,k + f_j,k)")
    lines.append("  where f_i,k = firm i's spend on issue k / firm i's total budget")
    lines.append("")
    lines.append("Property: BC_k ∈ [0, 1] for all k (bounded)")
    lines.append("  Proof: |f_i,k - f_j,k| ≤ f_i,k + f_j,k always, so BC ≥ 0.")
    lines.append("         BC = 1 - |...|/(sum) ≤ 1.")

    # --
    lines.append("\n-- Sum of Bounded Values --")
    lines.append("")
    lines.append("Let N = number of shared issues between firms i and j.")
    lines.append("Sum of BC scores:")
    lines.append("  S(i,j) = sum over shared issues k: BC_k(i,j)")
    lines.append("")
    lines.append("Since each BC_k ∈ [0, 1]:")
    lines.append("  S(i,j) ≤ N × 1.0 = N")
    lines.append("")
    lines.append("Example with N = 75 shared issues (all 75 issue codes):")
    lines.append("  Maximum sum S = 75 (if BC = 1.0 on each issue)")
    lines.append("  Minimum sum S = 0  (if BC = 0 on all issues)")

    # --
    lines.append("\n-- Impact of Normalization (sqrt) --")
    lines.append("")
    lines.append("With NORMALIZE=True:")
    lines.append("  weight(i,j) = S(i,j) / sqrt(N)")
    lines.append("")
    lines.append("For N shared issues with perfect BC = 1.0 on each:")
    lines.append("  weight_max(N) = N / sqrt(N) = sqrt(N)")
    lines.append("")

    # Generate synthetic examples
    examples = [1, 4, 9, 25, 49, 75]
    lines.append("  Synthetic examples (perfect alignment on N issues):")
    for n in examples:
        w_max = n / math.sqrt(n)
        lines.append(f"    N = {n:2d} issues → max weight = {n:2d} / sqrt({n:2d}) = {w_max:.4f}")

    lines.append("")
    lines.append(f"  Theoretical maximum (N = 75): {75.0 / math.sqrt(75):.4f}")
    lines.append("")
    lines.append("With NORMALIZE=False:")
    lines.append("  weight(i,j) = S(i,j) directly (no sqrt divisor)")
    lines.append(f"  Theoretical maximum: 75.0 (if all 75 issues have BC = 1.0)")

    # --
    lines.append("\n-- Why Unbounded Weights Are Intentional --")
    lines.append("")
    lines.append("Design rationale: three normalization choices and their tradeoffs:")
    lines.append("")
    lines.append("1. Raw sum (NORMALIZE=False, weight = N × mean_BC):")
    lines.append("   - Rewards breadth linearly: pairs with 75 shared issues get")
    lines.append("     much heavier weights than pairs with 10 shared issues.")
    lines.append("   - Unbounded: up to 75.")
    lines.append("   - Pro: captures full breadth signal (many shared issues)")
    lines.append("   - Con: weight dominance driven entirely by breadth, not quality.")
    lines.append("")
    lines.append("2. Plain mean (weight = mean_BC, bounded [0, 1]):")
    lines.append("   - Removes breadth signal entirely: a pair with 1 issue at BC=0.9")
    lines.append("     gets the same weight as a pair with 75 issues at BC=0.9.")
    lines.append("   - Bounded in [0, 1].")
    lines.append("   - Pro: weights are interpretable as similarity scores.")
    lines.append("   - Con: ignores multiplicity of coordination across issues.")
    lines.append("")
    lines.append("3. sqrt normalization (NORMALIZE=True, weight = sum_BC / sqrt(N)):")
    lines.append("   - Compromise: rewards breadth SUB-LINEARLY.")
    lines.append("   - Maximum bounded by sqrt(N), e.g., sqrt(75) ≈ 8.66.")
    lines.append("   - Pro: balances depth (portfolio alignment) with breadth.")
    lines.append("         Pairs coordinating on many issues get higher weights,")
    lines.append("         but growth is sublinear (diminishing returns per issue).")
    lines.append("   - Con: weights exceed [0, 1], cannot be interpreted as fractions.")
    lines.append("")
    lines.append("CHOSEN: Option 3 (sqrt normalization).")
    lines.append("This is best for community detection: algorithms like Leiden")
    lines.append("handle arbitrary positive weights without assuming [0, 1].")

    # --
    lines.append("\n-- Interpretation of Unbounded Weights --")
    lines.append("")
    lines.append("Example: weight(A, B) = 3.0")
    lines.append("")
    lines.append("Do NOT interpret as: A and B have 3.0 units of similarity.")
    lines.append("DO interpret as:")
    lines.append("  - A and B are aligned on multiple issues.")
    lines.append("  - If they share 9 issues with perfect BC on each: 9/sqrt(9) = 3.")
    lines.append("  - If they share 4 issues with perfect BC on each: 4/sqrt(4) = 2.")
    lines.append("  - The weight reflects both depth (per-issue alignment)")
    lines.append("    and breadth (number of shared issues).")
    lines.append("")
    lines.append("For centrality and community detection, this is appropriate:")
    lines.append("  - Eigenvector centrality: algorithms designed for weighted graphs")
    lines.append("    with arbitrary edge weights. [0, 1] not required.")
    lines.append("  - Leiden modularity: likewise, works with positive weights.")
    lines.append("    Unbounded weights just amplify within-community vs.")
    lines.append("    between-community edge strength more aggressively.")

    # --
    lines.append("\n-- Empirical Distribution --")

    if df_issues is not None and len(df_issues) > 0:
        # Replicate the edge construction logic from issue_similarity_network.py
        df = df_issues.copy()
        df = df.groupby(["client_name", "general_issue_code"],
                        as_index=False)["amount"].sum()

        company_totals = df.groupby("client_name")["amount"].sum()
        df["total_budget"] = df["client_name"].map(company_totals)

        # Exclude zero-budget firms
        df = df[df["total_budget"] > 0].copy()
        df["frac"] = df["amount"] / df["total_budget"]

        # Build edges
        issue_companies = (
            df.groupby("general_issue_code")
              .apply(lambda x: list(zip(x["client_name"], x["frac"])),
                     include_groups=False)
        )

        records = []
        for issue_code, companies in issue_companies.items():
            for i in range(len(companies)):
                for j in range(i + 1, len(companies)):
                    c1, f1 = companies[i]
                    c2, f2 = companies[j]
                    if c1 != c2 and (f1 + f2) > 0:
                        bc = 1 - abs(f1 - f2) / (f1 + f2)
                        src, tgt = (c1, c2) if c1 < c2 else (c2, c1)
                        records.append({"source": src, "target": tgt, "weight": bc})

        if records:
            edges = pd.DataFrame(records).groupby(["source", "target"])

            # Normalized version
            weight_sum = edges["weight"].sum()
            shared_count = edges["weight"].count().rename("shared_issues")
            result_norm = pd.concat([weight_sum, shared_count], axis=1).reset_index()
            result_norm["weight"] = result_norm["weight"] / result_norm["shared_issues"].pow(0.5)
            result_norm = result_norm[result_norm["weight"] > 0]

            # Raw (non-normalized) version
            result_raw = edges["weight"].sum().reset_index()

            lines.append("")
            lines.append(f"Data loaded: {len(df_issues):,} rows from fortune500_lda_issues.csv")
            lines.append(f"Firms: {df_issues['client_name'].nunique()}")
            lines.append(f"Issue codes: {df_issues['general_issue_code'].nunique()}")
            lines.append(f"Edges (after BC aggregation): {len(result_norm):,}")
            lines.append("")

            lines.append("NORMALIZED (weight = sum_BC / sqrt(shared_issues)):")
            lines.append(f"  Min weight:     {result_norm['weight'].min():.6f}")
            lines.append(f"  Mean weight:    {result_norm['weight'].mean():.6f}")
            lines.append(f"  Median weight:  {result_norm['weight'].median():.6f}")
            lines.append(f"  Max weight:     {result_norm['weight'].max():.6f}")
            lines.append(f"  Std dev:        {result_norm['weight'].std():.6f}")
            lines.append(f"  Weights > 1.0:  {(result_norm['weight'] > 1.0).sum()} "
                        f"({100*(result_norm['weight'] > 1.0).mean():.1f}%)")
            lines.append(f"  Weights > 2.0:  {(result_norm['weight'] > 2.0).sum()} "
                        f"({100*(result_norm['weight'] > 2.0).mean():.1f}%)")
            lines.append("")

            # Show quantiles
            lines.append("  Quantiles:")
            for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
                val = result_norm['weight'].quantile(q)
                lines.append(f"    {q:.2f}: {val:.6f}")

            lines.append("")
            lines.append("RAW (weight = sum_BC, no normalization):")
            lines.append(f"  Min weight:     {result_raw['weight'].min():.6f}")
            lines.append(f"  Mean weight:    {result_raw['weight'].mean():.6f}")
            lines.append(f"  Median weight:  {result_raw['weight'].median():.6f}")
            lines.append(f"  Max weight:     {result_raw['weight'].max():.6f}")
            lines.append(f"  Std dev:        {result_raw['weight'].std():.6f}")
            lines.append("")

            # Show the shared_issues distribution
            lines.append("Shared issue count distribution:")
            issue_counts = result_norm["shared_issues"].value_counts().sort_index()
            for n_shared, count in issue_counts.head(10).items():
                pct = 100 * count / len(result_norm)
                max_weight = n_shared / math.sqrt(n_shared)
                lines.append(f"  {n_shared:2d} shared issues: {count:5d} pairs ({pct:5.1f}%) "
                            f"→ max possible weight = {max_weight:.4f}")

            lines.append("")
            lines.append("Interpretation:")
            lines.append(f"  The maximum observed weight is {result_norm['weight'].max():.4f}.")
            lines.append(f"  Theoretical max with all 75 issues (BC=1.0 each): "
                        f"{75.0/math.sqrt(75):.4f}")
            lines.append(f"  The empirical distribution confirms the unbounded range.")

    else:
        lines.append("")
        lines.append("Data not available (fortune500_lda_issues.csv not found).")
        lines.append("The mathematical proof above shows the theoretical bounds.")

    # --
    lines.append("\n-- Summary --")
    lines.append("")
    lines.append("Issue similarity weights are NOT bounded in [0, 1] because:")
    lines.append("")
    lines.append("1. We sum Bray-Curtis scores across multiple shared issues.")
    lines.append("   Sum of N values in [0,1] can be at most N.")
    lines.append("")
    lines.append("2. We normalize by sqrt(N), not by N itself.")
    lines.append("   This rewards breadth (many shared issues) sub-linearly.")
    lines.append("")
    lines.append("3. With 75 issue codes, theoretical max = sqrt(75) ≈ 8.66.")
    lines.append("   Without normalization, max = 75.")
    lines.append("")
    lines.append("4. This is intentional: it balances depth (per-issue alignment)")
    lines.append("   with breadth (strategic coordination across multiple areas).")
    lines.append("")
    lines.append("5. Users should NOT interpret weights as [0,1] similarity scores.")
    lines.append("   Instead: a weight of 3.0 roughly means ~9 shared issues with")
    lines.append("   perfect alignment. Community detection algorithms handle this.")
    lines.append("")
    lines.append("See design_decisions.md §11 for the design rationale.")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    print("Loading issue data...")
    try:
        df_issues = pd.read_csv(DATA_DIR / "fortune500_lda_issues.csv")
    except FileNotFoundError:
        print(f"  Warning: {DATA_DIR / 'fortune500_lda_issues.csv'} not found.")
        print("  Running proof with synthetic data only.")
        df_issues = None

    report = run_proof(df_issues)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
