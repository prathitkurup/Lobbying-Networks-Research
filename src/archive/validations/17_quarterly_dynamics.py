"""
Quarterly dynamics of the directed influence network — 116th Congress (Q1–Q8).

Answers: Is net influencer status a stable within-congress property, or does the
set of top-10 agenda-setters fluctuate across the 8 quarterly windows?

Method:
  - Recompute net_influence per firm per quarter using the same RBO directed
    influence pipeline as rbo_directed_influence.py, but restricted to each
    quarter's lobbying data (report_type prefix matching).
  - Quarter mapping: 2019 Q1-4 → 1-4, 2020 Q1-4 → 5-8.
  - Prevalence filter (MAX_BILL_DF=50) applied per-quarter.
  - Global first-quarter lookup from the full congress is NOT used here; within
    each quarter, temporal precedence is determined by within-quarter first-mover
    status (which quarter the bill was first filed within the window). Since all
    filings in a given quarter share the same quarter index, within-quarter
    first-mover ties are broken by treating all as simultaneous (balanced).
    See design note in §17 below.
  - Outputs:
      (A) Bump chart: rank trajectories of top-10 overall firms across Q1–Q8.
          "Top-10 overall" = firms ranked in top-10 by net_influence in ≥3 quarters.
      (B) Heatmap: net_influence z-score per firm per quarter (firms = top overall,
          quarters = Q1-Q8).
      (C) Jaccard similarity of top-10 set between adjacent quarters.
      (D) Spearman rho of net_influence ranks between adjacent quarters.

Note on within-quarter first-mover: within a single quarter, all filings for a
(firm, bill) pair share the same quarter number. So `bill_first` computed within
just one quarter's data will yield all ties (every firm's earliest quarter = the
same constant). To resolve within-quarter temporal precedence we use report_type
ordering: q1 < q1a < q2 < ... within that quarter (amendment reports are later).
Firms with only amendment filings for a bill lose the first-mover to those with
base reports. This mirrors the global approach but restricted to one quarter.

Run from src/ directory:
  python validations/17_quarterly_dynamics.py
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import (
    aggregate_per_firm_bill, compute_zero_budget_fracs,
    build_ranked_lists, rbo_score,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR   = ROOT / "outputs" / "validation"
FIG_DIR   = ROOT / "visualizations" / "png"
TXT_PATH  = OUT_DIR / "17_quarterly_dynamics.txt"
CSV_NI    = OUT_DIR / "17_quarterly_net_influence.csv"   # firm × quarter net_influence
CSV_STAB  = OUT_DIR / "17_quarterly_stability.csv"       # Jaccard + Spearman per adjacent pair

CONGRESSES_USED = [116]
QUARTERS        = list(range(1, 9))       # Q1–Q8
QUARTER_LABELS  = [
    "Q1'19", "Q2'19", "Q3'19", "Q4'19",
    "Q1'20", "Q2'20", "Q3'20", "Q4'20",
]
TOP_N           = 10    # firms in leaderboard / bump chart
MIN_QUARTERS    = 3     # minimum quarters in top-10 to appear on bump chart
RBO_P           = 0.85
TOP_BILLS       = 30

# Report-type ordinal: lower = earlier within the quarter.
REPORT_ORDINAL = {
    "q1": 0, "q1t": 1, "q1a": 2, "q1ta": 3,
    "q2": 0, "q2t": 1, "q2a": 2, "q2ta": 3,
    "q3": 0, "q3t": 1, "q3a": 2,
    "q4": 0, "q4t": 1, "q4a": 2,
}


# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    """Write to stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()


# ---------------------------------------------------------------------------
# Quarter assignment
# ---------------------------------------------------------------------------

def assign_quarters(df):
    """Add 'quarter' (1–8) and 'report_ordinal' columns; no-op if already present."""
    df = df.copy()
    if "quarter" not in df.columns:
        base_q    = df["report_type"].str[1].astype(int)
        year_off  = df["year"].map({2019: 0, 2020: 4})
        df["quarter"] = base_q + year_off
    df["report_ordinal"] = df["report_type"].map(REPORT_ORDINAL).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Within-quarter first-mover lookup
# ---------------------------------------------------------------------------

def build_within_quarter_first(df_q):
    """
    Return {(firm, bill): report_ordinal_min} for a single quarter's dataframe.
    Lower ordinal = earlier within the quarter (base report before amendment).
    """
    return (
        df_q.groupby(["fortune_name", "bill_number"])["report_ordinal"]
        .min()
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Pairwise scoring (identical to rbo_directed_influence.py)
# ---------------------------------------------------------------------------

def score_pair(firm_a, firm_b, shared_bills, bill_first):
    """Tally within-quarter first-mover wins for a firm pair over shared top-30 bills."""
    a_firsts = b_firsts = ties = 0
    for bill in shared_bills:
        qa = bill_first.get((firm_a, bill))
        qb = bill_first.get((firm_b, bill))
        if qa is None or qb is None:
            continue
        if qa < qb:
            a_firsts += 1
        elif qb < qa:
            b_firsts += 1
        else:
            ties += 1
    return {"a_firsts": a_firsts, "b_firsts": b_firsts, "tie_count": ties}


# ---------------------------------------------------------------------------
# Per-quarter net_influence computation
# ---------------------------------------------------------------------------

def compute_quarter_net_influence(df_raw, quarter):
    """
    Compute net_influence per firm for one quarter of the 116th Congress.

    Steps mirror rbo_directed_influence.py:
      1. Filter to this quarter's filings.
      2. Aggregate per (firm, bill), compute fracs.
      3. Prevalence-filter bills (MAX_BILL_DF applied per quarter).
      4. Build top-30 ranked lists by spend fraction.
      5. Within-quarter first-mover lookup.
      6. Score all RBO-linked pairs; tally net_influence.

    Returns dict {firm: net_influence}.
    """
    df_q = df_raw[df_raw["quarter"] == quarter].copy()
    if df_q.empty:
        return {}

    df_agg = aggregate_per_firm_bill(df_q)
    df_agg = compute_zero_budget_fracs(df_agg)
    if MAX_BILL_DF is not None:
        df_agg = filter_bills_by_prevalence(df_agg, MAX_BILL_DF, unit_col="bill_number")

    ranked = build_ranked_lists(df_agg, top_bills=TOP_BILLS)
    if len(ranked) < 2:
        return {}

    bill_first = build_within_quarter_first(df_q)
    firms      = sorted(ranked.keys())

    # Tally first-mover counts per firm
    firsts  = {f: 0 for f in firms}
    losses  = {f: 0 for f in firms}

    for firm_a, firm_b in itertools.combinations(firms, 2):
        list_a = ranked[firm_a]
        list_b = ranked[firm_b]
        rbo_w  = rbo_score(list_a, list_b, p=RBO_P)
        if rbo_w == 0.0:
            continue

        shared = set(list_a) & set(list_b)
        sc     = score_pair(firm_a, firm_b, shared, bill_first)
        a_f, b_f = sc["a_firsts"], sc["b_firsts"]

        if a_f == b_f:
            continue   # tied pairs (net_temporal = 0) contribute 0 net
        if a_f > b_f:
            firsts[firm_a]  += a_f
            losses[firm_a]  += b_f
            firsts[firm_b]  += b_f
            losses[firm_b]  += a_f
        else:
            firsts[firm_b]  += b_f
            losses[firm_b]  += a_f
            firsts[firm_a]  += a_f
            losses[firm_a]  += b_f

    return {f: firsts[f] - losses[f] for f in firms}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def make_bump_chart(ni_wide, top_firms, out_path):
    """
    Bump chart: rank trajectories of top firms across Q1–Q8.
    Rank 1 = highest net_influence; lower y = better rank (y-axis inverted).
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(top_firms), 1))

    # Keep NaN where a firm is absent from a quarter; use float ranks
    rank_wide = ni_wide.rank(ascending=False, method="min")

    handles = []
    for i, firm in enumerate(top_firms):
        color = cmap(i)
        ranks = [rank_wide.loc[firm, q] if firm in rank_wide.index and q in rank_wide.columns
                 else np.nan for q in QUARTERS]
        ax.plot(QUARTER_LABELS, ranks, "o-", color=color, lw=2, ms=7, label=firm)
        # annotate final rank
        final_q = QUARTERS[-1]
        if firm in rank_wide.index and final_q in rank_wide.columns:
            r = rank_wide.loc[firm, final_q]
            if pd.notna(r):
                ax.annotate(
                    firm.split()[0][:12],
                    xy=(QUARTER_LABELS[-1], r),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=color, va="center",
                )
        handles.append(mpatches.Patch(color=color, label=firm[:30]))

    ax.invert_yaxis()
    max_rank = int(rank_wide.max().max()) if not rank_wide.empty else 12
    ax.set_yticks(range(1, max(max_rank + 2, 12)))
    ax.set_ylabel("Rank (net_influence, 1 = highest)", fontsize=11)
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_title(
        f"Top-{TOP_N} Net Influencer Rank Trajectories — 116th Congress (Q1–Q8)",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bump chart → {out_path.name}")


def make_heatmap(ni_wide, top_firms, out_path):
    """
    Heatmap of z-scored net_influence (rows = firms, cols = quarters).
    Z-score computed per quarter (column-wise) so ranks are comparable across time.
    """
    sub = ni_wide.loc[[f for f in top_firms if f in ni_wide.index], QUARTERS].copy()
    # Z-score per quarter
    zsub = (sub - sub.mean()) / sub.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_firms) * 0.45 + 1)))
    im = ax.imshow(zsub.values, aspect="auto", cmap="RdYlGn",
                   vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(len(QUARTERS)))
    ax.set_xticklabels(QUARTER_LABELS, fontsize=9)
    ax.set_yticks(range(len(top_firms)))
    ax.set_yticklabels(
        [f[:35] for f in [tf for tf in top_firms if tf in ni_wide.index]],
        fontsize=8,
    )

    # Annotate cells with raw net_influence values
    for ri, firm in enumerate([f for f in top_firms if f in ni_wide.index]):
        for ci, q in enumerate(QUARTERS):
            val = sub.loc[firm, q] if not pd.isna(sub.loc[firm, q]) else ""
            txt = f"{int(val):+d}" if val != "" else "—"
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=7,
                    color="black")

    plt.colorbar(im, ax=ax, label="Z-score (per quarter)", shrink=0.8)
    ax.set_title(
        f"Net Influence Heatmap — Top-{TOP_N} Firms × 8 Quarters (116th Congress)",
        fontsize=12,
    )
    ax.set_xlabel("Quarter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 72)
    print("VALIDATION 17: QUARTERLY DYNAMICS — 116th CONGRESS (Q1–Q8)")
    print("=" * 72)
    print(f"\nRBO_P={RBO_P}, TOP_BILLS={TOP_BILLS}, MAX_BILL_DF={MAX_BILL_DF}")
    print("Within-quarter first-mover: report_type ordinal (base < amendment)")
    print("Balanced pairs (tied first-mover) excluded from net_influence.")

    # -- Load data ------------------------------------------------------------
    print("\n[1/4] Loading 116th Congress reports ...")
    # Use the congress-specific reports (path mirrors multi_congress_pipeline)
    df_raw = pd.read_csv(DATA_DIR / "congress" / "116" / "opensecrets_lda_reports.csv")
    df_raw = assign_quarters(df_raw)
    print(f"  {len(df_raw):,} rows  |  {df_raw['fortune_name'].nunique()} firms")
    for q in QUARTERS:
        nf = df_raw[df_raw["quarter"] == q]["fortune_name"].nunique()
        print(f"  Q{q}: {nf} firms")

    # -- Per-quarter net_influence -------------------------------------------
    print("\n[2/4] Computing per-quarter net_influence (Q1–Q8) ...")
    ni_records = []
    for q in QUARTERS:
        ni = compute_quarter_net_influence(df_raw, q)
        n_firms = len(ni)
        n_pos   = sum(1 for v in ni.values() if v > 0)
        print(f"  Q{q}: {n_firms} firms  ({n_pos} with net_influence > 0)")
        for firm, val in ni.items():
            ni_records.append({"firm": firm, "quarter": q, "net_influence": val})

    long_df  = pd.DataFrame(ni_records)
    ni_wide  = long_df.pivot(index="firm", columns="quarter", values="net_influence")

    # Save full table
    ni_wide.reset_index().to_csv(CSV_NI, index=False)
    print(f"\n  Full net_influence table ({ni_wide.shape[0]} firms × 8 quarters) → {CSV_NI.name}")

    # -- Stability metrics ---------------------------------------------------
    print("\n[3/4] Stability: adjacent-quarter Jaccard (top-10) and Spearman rho ...")

    stab_rows = []
    print(f"\n  {'Pair':<10} {'|top10∩|':>9} {'Jaccard':>9} {'Spearman ρ':>12} {'p':>8}")
    print(f"  {'-'*55}")

    for i in range(len(QUARTERS) - 1):
        q1, q2 = QUARTERS[i], QUARTERS[i + 1]

        # Top-10 per quarter
        top10_q1 = set(
            long_df[long_df["quarter"] == q1]
            .nlargest(TOP_N, "net_influence")["firm"]
        )
        top10_q2 = set(
            long_df[long_df["quarter"] == q2]
            .nlargest(TOP_N, "net_influence")["firm"]
        )
        inter    = len(top10_q1 & top10_q2)
        union    = len(top10_q1 | top10_q2)
        jaccard  = inter / union if union > 0 else np.nan

        # Spearman on common firms
        common = ni_wide[[q1, q2]].dropna()
        if len(common) >= 5:
            rho, pval = spearmanr(common[q1], common[q2])
        else:
            rho = pval = np.nan

        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "" if not np.isnan(pval) else ""
        print(f"  Q{q1}→Q{q2}    {inter:>9} {jaccard:>9.3f} {rho:>12.3f} {pval:>8.4f} {stars}")

        stab_rows.append({
            "q_from": q1, "q_to": q2,
            "label_from": QUARTER_LABELS[i], "label_to": QUARTER_LABELS[i + 1],
            "top10_intersection": inter,
            "jaccard": round(jaccard, 4),
            "spearman_rho": round(rho, 4),
            "spearman_pval": round(pval, 5),
        })

    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(CSV_STAB, index=False)
    print(f"\n  Stability CSV → {CSV_STAB.name}")

    # Summary stats
    mean_j   = stab_df["jaccard"].mean()
    mean_rho = stab_df["spearman_rho"].mean()
    print(f"\n  Mean Jaccard (adjacent top-10 overlap): {mean_j:.3f}")
    print(f"  Mean Spearman ρ (adjacent rank corr):   {mean_rho:.3f}")

    # -- Identify focal firms for figures ------------------------------------
    # Top-MIN_QUARTERS approach: firms in top-10 in ≥ MIN_QUARTERS quarters
    top10_counts = {}
    for q in QUARTERS:
        for firm in long_df[long_df["quarter"] == q].nlargest(TOP_N, "net_influence")["firm"]:
            top10_counts[firm] = top10_counts.get(firm, 0) + 1

    focal_firms = sorted(
        [f for f, cnt in top10_counts.items() if cnt >= MIN_QUARTERS],
        key=lambda f: -top10_counts[f]
    )[:20]   # cap at 20 for readability

    print(f"\n  Firms in top-10 in ≥{MIN_QUARTERS} quarters: {len(focal_firms)}")
    print(f"\n  {'Firm':<45} {'Quarters in top-10':>20}")
    print(f"  {'-'*67}")
    for f in focal_firms:
        quarters_in = [
            QUARTER_LABELS[q - 1]
            for q in QUARTERS
            if f in [
                row["firm"]
                for _, row in long_df[long_df["quarter"] == q]
                .nlargest(TOP_N, "net_influence")
                .iterrows()
            ]
        ]
        print(f"  {f:<45} {top10_counts[f]:>3}/8  ({', '.join(quarters_in)})")

    # -- Figures -------------------------------------------------------------
    print("\n[4/4] Generating figures ...")

    make_bump_chart(
        ni_wide, focal_firms,
        FIG_DIR / "17_bump_chart_quarterly_influencers.png",
    )
    make_heatmap(
        ni_wide, focal_firms,
        FIG_DIR / "17_heatmap_quarterly_net_influence.png",
    )

    print(f"\n  Stability CSV         → {CSV_STAB}")
    print(f"  Net influence table   → {CSV_NI}")
    print(f"  Log                   → {TXT_PATH}")
    print("\n  Validation complete.")
    print("=" * 72)

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
