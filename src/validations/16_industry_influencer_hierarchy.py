"""
Industry-sector influencer hierarchy (111th-117th Congress).

Answers two questions per Leiden affiliation community:
  (1) Who are the top within-community agenda-setters in each congress?
  (2) Has that ranking stayed stable across sessions?

Communities (116th-Congress affiliation Leiden partition):
  0: Finance/Insurance   (n=72)
  1: Tech/Telecom        (n=64)
  2: Defense/Industrial  (n=49)
  3: Energy/Utilities    (n=49)
  4: Health/Pharma       (n=43)

Three analyses per community:
  A. Top-5 within-community agenda-setters by net_influence, per congress.
     "Within-community" means net_influence is computed only from directed
     edges where both endpoints share the same affiliation community — this
     isolates intra-sector agenda-setting from cross-sector noise.
  B. Within-community rank stability: adjacent-congress Spearman rho on
     within-community net_influence ranks, and Kendall's W across all
     sessions. Stable-set firms only (present in all 7 congresses).
  C. Persistent leaders: firms that appear in the top-5 within-community
     in at least 4 of 7 congresses.

Within-community net_influence is computed fresh from
data/congress/{num}/rbo_directed_influence.csv for each congress,
restricted to edges where both source and target are in the same
affiliation community.

Run from src/ directory:
  python validations/16_industry_influencer_hierarchy.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR   = ROOT / "outputs" / "validation"
TXT_PATH  = OUT_DIR / "16_industry_influencer_hierarchy.txt"
CSV_RANKS = OUT_DIR / "16_within_community_ni_by_congress.csv"
CSV_STAB  = OUT_DIR / "16_within_community_rank_stability.csv"

COMMUNITY_LABELS = {
    0: "Finance/Insurance",
    1: "Tech/Telecom",
    2: "Defense/Industrial",
    3: "Energy/Utilities",
    4: "Health/Pharma",
}

CONGRESSES = [111, 112, 113, 114, 115, 116, 117]
TOP_N      = 5     # firms shown in leaderboard per community per congress
MIN_SESSIONS_W = 3 # minimum non-NaN sessions for Kendall's W inclusion


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
# Within-community net_influence
# ---------------------------------------------------------------------------

def compute_wc_net_influence(congress, comm_map):
    """
    Compute within-community net_influence for every firm in a given congress.
    Restricts directed edges to those where both endpoints share the same
    affiliation community label. Returns {firm: wc_net_influence}.
    """
    path = DATA_DIR / f"congress/{congress}/rbo_directed_influence.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    directed = df[df["balanced"] == 0].copy()
    directed["src_comm"] = directed["source"].map(comm_map)
    directed["tgt_comm"] = directed["target"].map(comm_map)

    # Keep only intra-community edges with known communities
    intra = directed[
        directed["src_comm"].notna() &
        directed["tgt_comm"].notna() &
        (directed["src_comm"] == directed["tgt_comm"])
    ]

    as_src = intra.groupby("source").agg(
        out_firsts  = ("source_firsts", "sum"),
        out_losses  = ("target_firsts", "sum"),
    ).rename_axis("firm")

    as_tgt = intra.groupby("target").agg(
        in_wins     = ("target_firsts", "sum"),
        in_losses   = ("source_firsts", "sum"),
    ).rename_axis("firm")

    merged = as_src.join(as_tgt, how="outer").fillna(0)
    merged["wc_net_influence"] = (
        merged["out_firsts"] + merged["in_wins"]
        - merged["out_losses"] - merged["in_losses"]
    )
    return merged["wc_net_influence"].to_dict()


# ---------------------------------------------------------------------------
# Kendall's W
# ---------------------------------------------------------------------------

def kendalls_w(rank_matrix):
    """
    Kendall's W from an (n_raters x n_subjects) rank matrix with possible NaNs.
    Subjects with fewer than MIN_SESSIONS_W non-NaN ratings are excluded.
    Returns (W, chi2, p_approx) or (nan, nan, nan) if insufficient data.
    """
    m_full = rank_matrix.copy()
    # Drop columns (firms) with too few ratings
    m_full = m_full.loc[:, m_full.notna().sum(axis=0) >= MIN_SESSIONS_W]
    if m_full.shape[1] < 2:
        return np.nan, np.nan, np.nan

    # Rank within each row (congress), ignoring NaN
    ranked = m_full.apply(lambda row: row.rank(method="average", na_option="keep"), axis=1)

    n = ranked.shape[1]     # subjects (firms)
    k = ranked.shape[0]     # raters  (congresses)

    # Sum of ranks per subject
    R = ranked.sum(axis=0, skipna=True)
    R_bar = R.mean()
    S = ((R - R_bar) ** 2).sum()

    W = 12 * S / (k ** 2 * (n ** 3 - n))
    W = float(np.clip(W, 0, 1))

    # Chi-squared approximation (Siegel & Castellan 1988)
    chi2 = k * (n - 1) * W
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=n - 1)
    return round(W, 4), round(chi2, 2), round(p, 5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 72)
    print("VALIDATION 16: WITHIN-COMMUNITY INFLUENCER HIERARCHY (111th-117th)")
    print("=" * 72)
    print("\nCommunity labels (Leiden, 116th affiliation network):")
    for k, v in COMMUNITY_LABELS.items():
        print(f"  {k}: {v}")
    print(f"\nTop-N leaderboard: {TOP_N} firms per community per congress")
    print("Within-community net_influence: intra-sector directed edges only.")

    # Load community partition
    comm_df  = pd.read_csv(DATA_DIR / "archive" / "communities" / "communities_affiliation.csv")
    comm_map = dict(zip(comm_df["fortune_name"], comm_df["community_aff"]))

    # Build full within-community net_influence table: firm x congress
    print("\n[1/3] Computing within-community net_influence for all congresses ...")
    all_records = []
    for congress in CONGRESSES:
        wc_ni = compute_wc_net_influence(congress, comm_map)
        for firm, val in wc_ni.items():
            all_records.append({
                "firm": firm, "congress": congress, "wc_net_influence": val
            })
        print(f"      {congress}th: {len(wc_ni)} firms with intra-community edges")

    long_df = pd.DataFrame(all_records)
    long_df["community"] = long_df["firm"].map(comm_map)
    long_df = long_df.dropna(subset=["community"])
    long_df["community"] = long_df["community"].astype(int)
    long_df["sector"] = long_df["community"].map(COMMUNITY_LABELS)

    # Wide form: firm x congress
    wide_df = long_df.pivot(index="firm", columns="congress", values="wc_net_influence")
    wide_df["community"] = wide_df.index.map(comm_map)
    wide_df = wide_df.dropna(subset=["community"])
    wide_df["community"] = wide_df["community"].astype(int)

    # Save full rank table
    wide_df.reset_index().to_csv(CSV_RANKS, index=False)

    # -----------------------------------------------------------------------
    # Per-community analyses
    # -----------------------------------------------------------------------

    stability_rows = []

    print("\n[2/3] Per-community leaderboards and rank stability ...")

    for c_id, c_label in COMMUNITY_LABELS.items():
        members = set(comm_df[comm_df["community_aff"] == c_id]["fortune_name"].tolist())
        comm_long = long_df[long_df["community"] == c_id]
        comm_wide = wide_df[wide_df["community"] == c_id].drop(columns="community")

        stable_firms = comm_wide.dropna()
        n_stable = len(stable_firms)

        print(f"\n{'='*68}")
        print(f"  Community {c_id}: {c_label}")
        print(f"  Total members: {len(members)} | "
              f"Stable across all 7 congresses: {n_stable}")
        print(f"{'='*68}")

        # A. Top-5 leaderboard per congress
        print(f"\n  Top-{TOP_N} within-community agenda-setters by congress:\n")

        # Header row
        cong_width = 32
        header = f"  {'Rank':<5}" + "".join(f"  {c}th{'':<{cong_width-4}}" for c in CONGRESSES)
        print(header)
        print(f"  {'-' * (5 + len(CONGRESSES) * (cong_width + 2))}")

        # Collect top-N per congress
        tops = {}
        for congress in CONGRESSES:
            sub = comm_long[comm_long["congress"] == congress].nlargest(TOP_N, "wc_net_influence")
            tops[congress] = list(zip(sub["firm"].tolist(), sub["wc_net_influence"].tolist()))

        for rank_i in range(TOP_N):
            row_str = f"  {rank_i+1:<5}"
            for congress in CONGRESSES:
                entry = tops[congress][rank_i] if rank_i < len(tops[congress]) else ("—", 0)
                # Truncate firm name to fit column
                name = entry[0][:22]
                val  = int(entry[1])
                cell = f"{name} ({val:+d})"
                row_str += f"  {cell:<{cong_width}}"
            print(row_str)

        # B. Within-community rank stability
        if n_stable >= 5:
            # Adjacent-congress Spearman on stable set
            adj_rhos = []
            for i in range(len(CONGRESSES) - 1):
                c1, c2 = CONGRESSES[i], CONGRESSES[i + 1]
                if c1 in stable_firms.columns and c2 in stable_firms.columns:
                    v1 = stable_firms[c1].dropna()
                    v2 = stable_firms[c2].dropna()
                    common = v1.index.intersection(v2.index)
                    if len(common) >= 5:
                        rho, pval = spearmanr(v1[common], v2[common])
                        adj_rhos.append((c1, c2, round(rho, 3), round(pval, 4)))

            # Kendall's W across all congresses on stable set
            rank_matrix = stable_firms[CONGRESSES].T  # shape: (n_congresses, n_firms)
            W, chi2_val, p_W = kendalls_w(rank_matrix)

            print(f"\n  Rank stability (within-community net_influence, n={n_stable} stable firms):")
            print(f"\n    Kendall's W = {W}  (chi2={chi2_val}, p={p_W})")
            print(f"\n    Adjacent-congress Spearman rho:")
            for c1, c2, rho, pval in adj_rhos:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                sig = "p<0.001" if pval < 0.001 else f"p={pval:.4f}"
                print(f"      {c1}→{c2}: rho={rho:+.3f}  {sig} {stars}")

            stability_rows.append({
                "community": c_id,
                "sector": c_label,
                "n_stable": n_stable,
                "kendalls_W": W,
                "kendalls_W_pval": p_W,
                **{f"rho_{r[0]}_{r[1]}": r[2] for r in adj_rhos},
                **{f"pval_{r[0]}_{r[1]}": r[3] for r in adj_rhos},
            })
        else:
            print(f"\n  Rank stability: insufficient stable firms (n={n_stable} < 5).")

        # C. Persistent leaders: appear in top-5 in >= 4 of 7 congresses
        appearance_counts = {}
        for congress in CONGRESSES:
            sub = comm_long[comm_long["congress"] == congress].nlargest(TOP_N, "wc_net_influence")
            for firm in sub["firm"]:
                appearance_counts[firm] = appearance_counts.get(firm, 0) + 1

        persistent = sorted(
            [(f, cnt) for f, cnt in appearance_counts.items() if cnt >= 4],
            key=lambda x: -x[1]
        )

        print(f"\n  Persistent leaders (top-{TOP_N} in ≥4 of 7 congresses):")
        if persistent:
            for firm, cnt in persistent:
                # Show their net_influence in each congress they appear in top-5
                in_congresses = [
                    str(c) for c in CONGRESSES
                    if firm in [x[0] for x in tops.get(c, [])]
                ]
                print(f"    {firm:<42} {cnt}/7 congresses  ({', '.join(in_congresses)}th)")
        else:
            print("    None — no firm reaches top-5 in ≥4 of 7 congresses.")

    # -----------------------------------------------------------------------
    # Summary stability table across communities
    # -----------------------------------------------------------------------

    if stability_rows:
        print(f"\n{'='*72}")
        print(f"  Summary: Kendall's W by community")
        print(f"{'='*72}")
        print(f"\n  {'Community':<22} {'n_stable':>9} {'W':>8} {'p':>10}  interpretation")
        print(f"  {'-'*65}")
        for r in stability_rows:
            W_val = r["kendalls_W"]
            p_val = r["kendalls_W_pval"]
            if pd.isna(W_val):
                interp = "insufficient data"
            elif p_val < 0.05 and W_val >= 0.3:
                interp = "significant concordance"
            elif p_val < 0.05:
                interp = "weak but significant"
            else:
                interp = "not significant"
            print(f"  {r['sector']:<22} {r['n_stable']:>9} {W_val:>8.4f} {p_val:>10.5f}  {interp}")

        stab_df = pd.DataFrame(stability_rows)
        stab_df.to_csv(CSV_STAB, index=False)

    print(f"\n  Rank table (firm × congress)  → {CSV_RANKS}")
    print(f"  Stability summary             → {CSV_STAB}")
    print(f"  Log                           → {TXT_PATH}")
    print("\n  Validation complete.")
    print("=" * 72)

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
