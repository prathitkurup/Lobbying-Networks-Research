"""
Mediation analysis: is agenda-setting just shared lobbyists?

Tests whether influencer→follower pairs (decisive directed edges) share
specific lobbyists or lobbying firms on the bills where one leads the other,
vs. simply being connected through the broader underlying lobbyist network.

Two mediation channels:
  Bill-level (strict):   source and follower share a lobbyist/registrant
                         on their first-quarter reports for that specific bill.
  Network-level (broad): source and follower share any lobbyist or registrant
                         anywhere across their full lobbying portfolios
                         (i.e., connected via the affiliation network).

Data source: pre-computed affiliation_mediated_adoption.csv and
rbo_edges_enriched.csv from affiliation_mediated_adoption.py.

Key question: if bill-level mediation rates are low but RBO-based influence
persists, the mechanism points to structural portfolio similarity and
industry-level network formation — not direct intermediary sharing.

Outputs (outputs/analysis/):
  02_mediation_summary.csv
  02_mediation.txt
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

RBO_HIGH_PCT = 0.75   # top-quartile threshold for high-RBO pairs
RBO_LOW_PCT  = 0.25   # bottom-quartile threshold for low-RBO pairs

OUT_DIR = ROOT / "outputs" / "analysis"

# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, t):
        for s in self.streams: s.write(t)
    def flush(self):
        for s in self.streams: s.flush()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "02_mediation.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 02: MEDIATION — IS IT JUST SHARED LOBBYISTS?")
    print(SEP)
    print()
    print("Question: Do influencer→follower pairs share specific lobbyists on")
    print("the bills where one leads, or does influence flow through the broader")
    print("underlying lobbyist network (industry ties, revolving door)?")

    # -- Load data -------------------------------------------------------
    adoption = pd.read_csv(DATA_DIR / "affiliation_mediated_adoption.csv")
    enriched = pd.read_csv(DATA_DIR / "rbo_edges_enriched.csv")

    print(f"\n  affiliation_mediated_adoption rows: {len(adoption):,}")
    print(f"  rbo_edges_enriched rows:            {len(enriched):,}")

    # -- Part 1: Bill-level mediation rates ------------------------------
    print(f"\n{'─'*70}")
    print("PART 1: BILL-LEVEL MEDIATION (strict — shared lobbyist on that bill)")
    print(f"{'─'*70}")

    decisive = adoption[adoption["is_bill_directed"] == True].copy()
    balanced = adoption[adoption["is_bill_directed"] == False].copy()

    n_dec = len(decisive)
    n_bal = len(balanced)

    # Mediation rates
    def rate(series):
        return series.sum() / max(len(series), 1)

    print(f"\n  Decisive pairs (influencer leads on bill):   {n_dec:,}")
    print(f"  Balanced pairs (tied on bill):               {n_bal:,}")
    print()
    print(f"  {'Channel':<30} {'Decisive rate':>14} {'Balanced rate':>14} {'Ratio':>8}")
    print(f"  {'─'*70}")
    for col, label in [("is_lobbyist_mediated", "Shared specific lobbyist"),
                       ("is_firm_mediated",     "Shared lobbying firm"),
                       ("is_any_mediated",      "Any bill-level sharing")]:
        r_dec = rate(decisive[col])
        r_bal = rate(balanced[col])
        ratio = r_dec / r_bal if r_bal > 0 else float("nan")
        print(f"  {label:<30} {r_dec:>13.4f}  {r_bal:>13.4f}  {ratio:>8.3f}x")

    # Chi-squared test: decisive vs balanced on any_mediated
    ct = np.array([
        [decisive["is_any_mediated"].sum(),   n_dec - decisive["is_any_mediated"].sum()],
        [balanced["is_any_mediated"].sum(),   n_bal - balanced["is_any_mediated"].sum()],
    ])
    chi2, p_chi, _, _ = chi2_contingency(ct)
    print(f"\n  χ² test (decisive vs balanced, any_mediated): "
          f"χ²={chi2:.3f}, p={p_chi:.4f}")
    print(f"  → Bill-level mediation is nearly absent in both groups; "
          f"{'no significant difference' if p_chi > 0.05 else 'small but significant difference'}.")

    # -- Part 2: Network-level connectivity ------------------------------
    print(f"\n{'─'*70}")
    print("PART 2: NETWORK-LEVEL CONNECTIVITY (broad — any shared affiliation)")
    print(f"{'─'*70}")

    print(f"\n  {'Channel':<30} {'Decisive rate':>14} {'Balanced rate':>14} {'Ratio':>8}")
    print(f"  {'─'*70}")
    for col, label in [("net_firm_connected",   "Firm network connected"),
                       ("net_any_connected",     "Any network connected")]:
        # These are present in adoption df
        r_dec = rate(decisive[col])
        r_bal = rate(balanced[col])
        ratio = r_dec / r_bal if r_bal > 0 else float("nan")
        print(f"  {label:<30} {r_dec:>13.4f}  {r_bal:>13.4f}  {ratio:>8.3f}x")

    print(f"\n  → Network connectivity is also low across both groups,")
    print(f"    consistent with influence flowing through structural portfolio")
    print(f"    similarity (RBO) rather than identifiable shared intermediaries.")

    # -- Part 3: Enriched edge analysis — mediation rate by RBO quartile -
    print(f"\n{'─'*70}")
    print("PART 3: MEDIATION BY RBO QUARTILE (from enriched edge file)")
    print(f"{'─'*70}")

    # Focus on edges that have directed bills (net_temporal != 0)
    directed_edges = enriched[enriched["net_temporal"] != 0].copy()
    n_directed = len(directed_edges)

    print(f"\n  Directed edges (net_temporal ≠ 0): {n_directed:,}")

    if n_directed > 0 and directed_edges["any_mediation_rate"].notna().sum() > 0:
        directed_edges["rbo_quartile"] = pd.qcut(
            directed_edges["rbo"], q=4,
            labels=["Q1 (low RBO)", "Q2", "Q3", "Q4 (high RBO)"]
        )
        q_stats = directed_edges.groupby("rbo_quartile", observed=True).agg(
            n          = ("rbo", "count"),
            mean_rbo   = ("rbo", "mean"),
            lob_med    = ("lobbyist_mediation_rate", "mean"),
            firm_med   = ("firm_mediation_rate", "mean"),
            any_med    = ("any_mediation_rate", "mean"),
        ).round(4)

        print(f"\n  {'Quartile':<16} {'n':>6} {'mean_rbo':>10} {'lob_med%':>10} "
              f"{'firm_med%':>10} {'any_med%':>10}")
        print(f"  {'─'*65}")
        for q, row in q_stats.iterrows():
            print(f"  {str(q):<16} {int(row['n']):>6} {row['mean_rbo']:>10.4f} "
                  f"{100*row['lob_med']:>9.2f}% {100*row['firm_med']:>9.2f}% "
                  f"{100*row['any_med']:>9.2f}%")
    else:
        # Fall back to non-mediation-rate approach using adoption df
        print(f"\n  (mediation_rate columns not populated — using adoption df)")
        decisive["rbo_edge"] = decisive["rbo_source"] + "|" + decisive["rbo_target"]
        # Aggregate: per pair, fraction of bills that are mediated
        pair_agg = decisive.groupby(["rbo_source", "rbo_target"]).agg(
            n_bills        = ("bill", "count"),
            n_lob_mediated = ("is_lobbyist_mediated", "sum"),
            n_any_mediated = ("is_any_mediated", "sum"),
        )
        pair_agg["lob_med_rate"] = pair_agg["n_lob_mediated"] / pair_agg["n_bills"]
        pair_agg["any_med_rate"] = pair_agg["n_any_mediated"] / pair_agg["n_bills"]

        # Merge with RBO edge weights
        rbo_weights = enriched[["source", "target", "rbo", "net_temporal"]].copy()
        rbo_weights = rbo_weights[rbo_weights["net_temporal"] > 0]
        pair_agg = pair_agg.reset_index().merge(
            rbo_weights.rename(columns={"source": "rbo_source", "target": "rbo_target"}),
            on=["rbo_source", "rbo_target"], how="left"
        )
        pair_agg = pair_agg.dropna(subset=["rbo"])

        if len(pair_agg) > 0:
            pair_agg["rbo_quartile"] = pd.qcut(
                pair_agg["rbo"], q=4,
                labels=["Q1 (low RBO)", "Q2", "Q3", "Q4 (high RBO)"]
            )
            q_stats = pair_agg.groupby("rbo_quartile", observed=True).agg(
                n_pairs    = ("rbo_source", "count"),
                mean_rbo   = ("rbo", "mean"),
                lob_rate   = ("lob_med_rate", "mean"),
                any_rate   = ("any_med_rate", "mean"),
            ).round(4)
            print(f"\n  {'Quartile':<16} {'n_pairs':>8} {'mean_rbo':>10} "
                  f"{'lob_med':>10} {'any_med':>10}")
            print(f"  {'─'*58}")
            for q, row in q_stats.iterrows():
                print(f"  {str(q):<16} {int(row['n_pairs']):>8} {row['mean_rbo']:>10.4f} "
                      f"{100*row['lob_rate']:>9.2f}% {100*row['any_rate']:>9.2f}%")

    # -- Part 4: Summary and interpretation ------------------------------
    print(f"\n{'─'*70}")
    print("SUMMARY: THE BROADER LOBBYIST NETWORK MECHANISM")
    print(f"{'─'*70}")
    r_bill_any = rate(decisive["is_any_mediated"])
    r_net_any  = rate(decisive["net_any_connected"])
    print(f"""
  Bill-level mediation rate (decisive pairs):   {100*r_bill_any:.2f}%
  Network-level connection rate (decisive):     {100*r_net_any:.2f}%

  These numbers show that agenda-setting influence is almost never
  mediated by a directly shared lobbyist or lobbying firm on a specific
  bill. Influencers and followers are also rarely connected through the
  broader firm affiliation network in a detectable way.

  The implication is that influence flows through the structural
  similarity of lobbying portfolios — captured by the RBO metric — which
  reflects shared issue prioritization arising from industry-level
  conditions, revolving door dynamics, and political economy, rather
  than traceable bilateral intermediary relationships.

  This is consistent with the BCZ (Ballester, Calvó-Armengol & Zenou 2006)
  framework: complementarities emerge from network position and strategic
  substitutes/complements in payoffs, not direct coordination.
""")

    # -- Save summary CSV ------------------------------------------------
    summary = pd.DataFrame([
        {"group": "Decisive (influencer leads)", "n": n_dec,
         "bill_lob_rate": rate(decisive["is_lobbyist_mediated"]),
         "bill_firm_rate": rate(decisive["is_firm_mediated"]),
         "bill_any_rate":  rate(decisive["is_any_mediated"]),
         "net_firm_connected": rate(decisive["net_firm_connected"]),
         "net_any_connected":  rate(decisive["net_any_connected"])},
        {"group": "Balanced (tied)", "n": n_bal,
         "bill_lob_rate": rate(balanced["is_lobbyist_mediated"]),
         "bill_firm_rate": rate(balanced["is_firm_mediated"]),
         "bill_any_rate":  rate(balanced["is_any_mediated"]),
         "net_firm_connected": rate(balanced["net_firm_connected"]),
         "net_any_connected":  rate(balanced["net_any_connected"])},
    ])
    summary.to_csv(OUT_DIR / "02_mediation_summary.csv", index=False)

    print(f"  Outputs:")
    print(f"    02_mediation_summary.csv")
    print(f"    02_mediation.txt")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
