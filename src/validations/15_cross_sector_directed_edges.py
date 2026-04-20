"""
Cross-sector directed edge analysis (116th Congress).

Tags each directed edge by whether source and target belong to the same
Leiden community (intra-sector) or different communities (cross-sector),
using the affiliation-network community partition.

Five analyses:
  1. Edge-level: cross-sector vs. intra-sector RBO weight and net_temporal
     distributions (Mann-Whitney U tests).
  2. Community-pair flow matrix: directed edge counts and mean RBO weight
     for every (src_community, tgt_community) pair; net directional flow
     (asymmetry) between community pairs.
  3. Firm-level cross-sector influence: net_cs_influence and net_cs_strength
     per firm; top cross-sector agenda-setters and followers.
  4. Bridge firm identification: firms with the highest ratio of cross-sector
     directed edges to total directed edges — structural cross-industry
     bridges.
  5. Issue-space profiles of top cross-sector dyads: top-10 cross-sector
     directed pairs by net_temporal; their dominant issue codes and
     whether they share issue space (cosine similarity).

Community labels:
  0 = Finance / Insurance    (n=72)
  1 = Tech / Telecom         (n=64)
  2 = Defense / Industrial   (n=49)
  3 = Energy / Utilities     (n=49)
  4 = Health / Pharma        (n=43)

Run from src/ directory:
  python validations/15_cross_sector_directed_edges.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR   = ROOT / "outputs" / "validation"
TXT_PATH  = OUT_DIR / "15_cross_sector_directed_edges.txt"
EDGE_CSV  = OUT_DIR / "15_cross_sector_edge_table.csv"
FIRM_CSV  = OUT_DIR / "15_cross_sector_firm_table.csv"
PAIR_CSV  = OUT_DIR / "15_cross_sector_pair_matrix.csv"

COMMUNITY_LABELS = {
    0: "Finance/Insurance",
    1: "Tech/Telecom",
    2: "Defense/Industrial",
    3: "Energy/Utilities",
    4: "Health/Pharma",
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
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and merge all required data frames."""
    directed_raw = pd.read_csv(DATA_DIR / "congress" / "116" / "rbo_directed_influence.csv")
    comm_df      = pd.read_csv(DATA_DIR / "archive" / "communities" / "communities_affiliation.csv")
    nodes        = pd.read_csv(DATA_DIR / "congress" / "116" / "node_attributes.csv")
    issues       = pd.read_csv(DATA_DIR / "congress" / "116" / "opensecrets_lda_issues.csv")
    return directed_raw, comm_df, nodes, issues

# ---------------------------------------------------------------------------
# Edge tagging
# ---------------------------------------------------------------------------

def tag_edges(directed_raw, comm_map):
    """
    Keep only directed (balanced=0) edges where both endpoints have community
    labels. Add src_comm, tgt_comm, cross_sector, community_pair columns.
    """
    df = directed_raw[directed_raw["balanced"] == 0].copy()
    df["src_comm"] = df["source"].map(comm_map)
    df["tgt_comm"] = df["target"].map(comm_map)
    df = df.dropna(subset=["src_comm", "tgt_comm"])
    df["src_comm"] = df["src_comm"].astype(int)
    df["tgt_comm"] = df["tgt_comm"].astype(int)
    df["cross_sector"] = df["src_comm"] != df["tgt_comm"]
    df["src_label"] = df["src_comm"].map(COMMUNITY_LABELS)
    df["tgt_label"] = df["tgt_comm"].map(COMMUNITY_LABELS)
    df["community_pair"] = df.apply(
        lambda r: f"{COMMUNITY_LABELS[r['src_comm']]} → {COMMUNITY_LABELS[r['tgt_comm']]}",
        axis=1,
    )
    return df

# ---------------------------------------------------------------------------
# Analysis 1: edge-level distributions
# ---------------------------------------------------------------------------

def analysis_edge_distributions(df):
    """Mann-Whitney U tests comparing RBO weight and net_temporal: cross vs intra."""
    cs   = df[df["cross_sector"]]
    intr = df[~df["cross_sector"]]

    results = {}
    for metric in ["weight", "net_temporal"]:
        stat, pval = mannwhitneyu(cs[metric], intr[metric], alternative="two-sided")
        results[metric] = {
            "cs_mean":   cs[metric].mean(),
            "cs_median": cs[metric].median(),
            "intra_mean":   intr[metric].mean(),
            "intra_median": intr[metric].median(),
            "mwu_stat": stat,
            "pval":      pval,
        }
    return results, cs, intr

# ---------------------------------------------------------------------------
# Analysis 2: community-pair flow matrix
# ---------------------------------------------------------------------------

def analysis_pair_matrix(df):
    """
    Directed edge counts, mean RBO weight, total net_temporal by community pair.
    Also compute net directional asymmetry between each unordered community pair:
    net_flow = edges(A→B) - edges(B→A).
    """
    pair_df = (df.groupby(["src_comm", "tgt_comm"])
                 .agg(
                     n_directed  = ("weight", "count"),
                     mean_weight = ("weight", "mean"),
                     sum_net_temporal = ("net_temporal", "sum"),
                 )
                 .reset_index())

    # Net directional asymmetry for each unordered pair
    asym_rows = []
    seen = set()
    for _, r in pair_df.iterrows():
        a, b = int(r["src_comm"]), int(r["tgt_comm"])
        if a == b or (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        ab = pair_df[(pair_df["src_comm"]==a) & (pair_df["tgt_comm"]==b)]["n_directed"].sum()
        ba = pair_df[(pair_df["src_comm"]==b) & (pair_df["tgt_comm"]==a)]["n_directed"].sum()
        dominant = COMMUNITY_LABELS[a] if ab >= ba else COMMUNITY_LABELS[b]
        asym_rows.append({
            "comm_a": COMMUNITY_LABELS[a],
            "comm_b": COMMUNITY_LABELS[b],
            "edges_a_to_b": int(ab),
            "edges_b_to_a": int(ba),
            "net_flow_a_minus_b": int(ab - ba),
            "dominant_sector": dominant,
        })

    pair_df["src_label"] = pair_df["src_comm"].map(COMMUNITY_LABELS)
    pair_df["tgt_label"] = pair_df["tgt_comm"].map(COMMUNITY_LABELS)
    return pair_df, pd.DataFrame(asym_rows).sort_values("net_flow_a_minus_b",
                                                         ascending=False)

# ---------------------------------------------------------------------------
# Analysis 3: firm-level cross-sector influence
# ---------------------------------------------------------------------------

def analysis_firm_cs_influence(df, nodes, comm_map):
    """Net cross-sector influence and strength per firm."""
    cs = df[df["cross_sector"]].copy()

    as_src = cs.groupby("source").agg(
        cs_out_edges    = ("weight", "count"),
        cs_out_firsts   = ("source_firsts", "sum"),
        cs_out_losses   = ("target_firsts", "sum"),
        cs_out_weight   = ("weight", "sum"),
    ).rename_axis("firm")

    as_tgt = cs.groupby("target").agg(
        cs_in_edges   = ("weight", "count"),
        cs_in_wins    = ("target_firsts", "sum"),
        cs_in_losses  = ("source_firsts", "sum"),
        cs_in_weight  = ("weight", "sum"),
    ).rename_axis("firm")

    firm_cs = as_src.join(as_tgt, how="outer").fillna(0)
    firm_cs["net_cs_influence"] = (
        firm_cs["cs_out_firsts"] + firm_cs["cs_in_wins"]
        - firm_cs["cs_out_losses"] - firm_cs["cs_in_losses"]
    )
    firm_cs["net_cs_strength"] = firm_cs["cs_out_weight"] - firm_cs["cs_in_weight"]
    firm_cs["community"] = pd.Series(firm_cs.index).map(comm_map).values
    firm_cs["community_label"] = firm_cs["community"].map(
        lambda x: COMMUNITY_LABELS.get(int(x), "unknown") if pd.notna(x) else "unknown"
    )

    nodes_idx = nodes.set_index("firm")
    firm_cs = firm_cs.join(nodes_idx[["net_influence", "net_strength"]], how="left")

    # Cross-sector share: how much of total influence comes from cross-sector edges
    firm_cs["cs_share"] = np.where(
        firm_cs["net_influence"].abs() > 0,
        firm_cs["net_cs_influence"] / firm_cs["net_influence"].abs(),
        np.nan,
    )

    return firm_cs.reset_index()

# ---------------------------------------------------------------------------
# Analysis 4: bridge firms
# ---------------------------------------------------------------------------

def analysis_bridge_firms(df, firm_cs_df):
    """
    Firms with highest cross-sector directed edge share.
    total_directed = cs_out_edges + cs_in_edges + intra-sector edges.
    """
    # Count total directed edges per firm (as source or target)
    all_dir = df.copy()
    total_out = all_dir.groupby("source").size().rename("total_out_edges")
    total_in  = all_dir.groupby("target").size().rename("total_in_edges")

    firm_total = pd.concat([total_out, total_in], axis=1).fillna(0)
    firm_total["total_edges"] = firm_total["total_out_edges"] + firm_total["total_in_edges"]
    firm_total = firm_total.rename_axis("firm").reset_index()

    merged = firm_cs_df.merge(firm_total, on="firm", how="left")
    merged["cs_total_edges"]  = merged["cs_out_edges"] + merged["cs_in_edges"]
    merged["cs_edge_fraction"] = merged["cs_total_edges"] / merged["total_edges"].clip(lower=1)

    # Bridge = high cross-sector fraction AND positive net_cs_influence
    return merged.sort_values("cs_edge_fraction", ascending=False)

# ---------------------------------------------------------------------------
# Analysis 5: top cross-sector dyads with issue profiles
# ---------------------------------------------------------------------------

def analysis_top_dyads(df, issues):
    """
    Top-10 cross-sector directed pairs by net_temporal.
    For each pair, compute the dominant issue codes for source and target
    and the cosine similarity of their issue-code profiles.
    """
    cs = df[df["cross_sector"]].copy()
    top_dyads = cs.nlargest(10, "net_temporal")[
        ["source", "target", "src_label", "tgt_label",
         "net_temporal", "weight", "source_firsts", "target_firsts"]
    ].reset_index(drop=True)

    # Build firm × issue_code spend matrix
    firm_issue = (issues.groupby(["fortune_name", "issue_code"])["amount_allocated"]
                        .sum()
                        .unstack(fill_value=0))

    rows = []
    for _, dyad in top_dyads.iterrows():
        src, tgt = dyad["source"], dyad["target"]
        src_issues = firm_issue.loc[src] if src in firm_issue.index else None
        tgt_issues = firm_issue.loc[tgt] if tgt in firm_issue.index else None

        if src_issues is not None and tgt_issues is not None:
            # Align on common columns
            all_codes = firm_issue.columns
            sv = src_issues.reindex(all_codes, fill_value=0).values.reshape(1, -1)
            tv = tgt_issues.reindex(all_codes, fill_value=0).values.reshape(1, -1)
            cos_sim = float(cosine_similarity(sv, tv)[0, 0])
            src_top3 = src_issues.nlargest(3).index.tolist()
            tgt_top3 = tgt_issues.nlargest(3).index.tolist()
            shared_codes = sorted(set(src_top3) & set(tgt_top3))
        else:
            cos_sim = np.nan
            src_top3 = tgt_top3 = shared_codes = []

        rows.append({
            "source":       src,
            "target":       tgt,
            "src_sector":   dyad["src_label"],
            "tgt_sector":   dyad["tgt_label"],
            "net_temporal": dyad["net_temporal"],
            "rbo_weight":   round(dyad["weight"], 5),
            "source_firsts": dyad["source_firsts"],
            "target_firsts": dyad["target_firsts"],
            "src_top3_issues": ", ".join(src_top3),
            "tgt_top3_issues": ", ".join(tgt_top3),
            "shared_top3_issues": ", ".join(shared_codes) if shared_codes else "none",
            "issue_cosine_sim": round(cos_sim, 4) if not np.isnan(cos_sim) else np.nan,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 72)
    print("VALIDATION 15: CROSS-SECTOR DIRECTED EDGE ANALYSIS (116th CONGRESS)")
    print("=" * 72)

    print("\nCommunity labels (Leiden, affiliation network):")
    for k, v in COMMUNITY_LABELS.items():
        print(f"  {k}: {v}")

    # -- Load ----------------------------------------------------------------
    print("\n[1/6] Loading data ...")
    directed_raw, comm_df, nodes, issues = load_data()
    comm_map = dict(zip(comm_df["fortune_name"], comm_df["community_aff"]))
    df = tag_edges(directed_raw, comm_map)
    print(f"      Directed edges with community labels: {len(df)}")
    print(f"      Cross-sector: {df['cross_sector'].sum()} "
          f"({df['cross_sector'].mean()*100:.1f}%)")
    print(f"      Intra-sector: {(~df['cross_sector']).sum()} "
          f"({(~df['cross_sector']).mean()*100:.1f}%)")

    # -- Analysis 1 ----------------------------------------------------------
    print("\n[2/6] Analysis 1: Edge-level distributions (cross vs. intra-sector)")
    dist_results, cs_edges, intra_edges = analysis_edge_distributions(df)

    for metric, r in dist_results.items():
        print(f"\n  {metric}:")
        print(f"    Cross-sector  — mean={r['cs_mean']:.5f},  median={r['cs_median']:.5f}")
        print(f"    Intra-sector  — mean={r['intra_mean']:.5f}, median={r['intra_median']:.5f}")
        print(f"    Mann-Whitney U={r['mwu_stat']:.1f}, p={r['pval']:.4e}")
        direction = "lower" if r["cs_mean"] < r["intra_mean"] else "higher"
        sig = "significant" if r["pval"] < 0.05 else "not significant"
        print(f"    → Cross-sector {metric} is {direction} ({sig})")

    # -- Analysis 2 ----------------------------------------------------------
    print("\n[3/6] Analysis 2: Community-pair flow matrix")
    pair_df, asym_df = analysis_pair_matrix(df)

    print("\n  Directed edge counts by community pair (rows=source, cols=target):")
    # Pivot for display
    pivot_n = pair_df.pivot(index="src_label", columns="tgt_label", values="n_directed").fillna(0).astype(int)
    print(pivot_n.to_string())

    print("\n  Mean RBO weight by community pair (rows=source, cols=target):")
    pivot_w = pair_df.pivot(index="src_label", columns="tgt_label", values="mean_weight").fillna(0).round(4)
    print(pivot_w.to_string())

    print("\n  Net directional flow between community pairs (positive = row dominates):")
    print(f"  {'Community A':<25} {'Community B':<25} {'A→B':>6} {'B→A':>6} "
          f"{'Net (A-B)':>10} {'Dominant'}")
    print(f"  {'-'*90}")
    for _, r in asym_df.iterrows():
        print(f"  {r['comm_a']:<25} {r['comm_b']:<25} "
              f"{r['edges_a_to_b']:>6} {r['edges_b_to_a']:>6} "
              f"{r['net_flow_a_minus_b']:>10} {r['dominant_sector']}")

    # -- Analysis 3 ----------------------------------------------------------
    print("\n[4/6] Analysis 3: Firm-level cross-sector influence")
    firm_cs = analysis_firm_cs_influence(df, nodes, comm_map)

    print(f"\n  Top 15 cross-sector agenda-setters (net_cs_influence):")
    print(f"  {'Firm':<42} {'Sector':<22} {'CS-NI':>6} {'CS-NS':>8} "
          f"{'NI':>6} {'CSout':>6} {'CSin':>5}")
    print(f"  {'-'*100}")
    top_setters = firm_cs.sort_values("net_cs_influence", ascending=False).head(15)
    for _, r in top_setters.iterrows():
        print(f"  {r['firm']:<42} {r['community_label']:<22} "
              f"{int(r['net_cs_influence']):>6} {r['net_cs_strength']:>8.3f} "
              f"{int(r['net_influence']) if pd.notna(r['net_influence']) else 'N/A':>6} "
              f"{int(r['cs_out_edges']):>6} {int(r['cs_in_edges']):>5}")

    print(f"\n  Top 10 cross-sector followers (most negative net_cs_influence):")
    print(f"  {'Firm':<42} {'Sector':<22} {'CS-NI':>6} {'NI':>6}")
    print(f"  {'-'*80}")
    top_followers = firm_cs.sort_values("net_cs_influence").head(10)
    for _, r in top_followers.iterrows():
        print(f"  {r['firm']:<42} {r['community_label']:<22} "
              f"{int(r['net_cs_influence']):>6} "
              f"{int(r['net_influence']) if pd.notna(r['net_influence']) else 'N/A':>6}")

    # Cross-sector influence by community
    print(f"\n  Mean net_cs_influence by community:")
    comm_summary = (firm_cs.dropna(subset=["community"])
                           .groupby("community_label")
                           .agg(
                               n_firms          = ("firm", "count"),
                               mean_cs_ni       = ("net_cs_influence", "mean"),
                               mean_cs_strength = ("net_cs_strength", "mean"),
                               pct_positive     = ("net_cs_influence",
                                                   lambda x: (x > 0).mean() * 100),
                           ))
    for label, row in comm_summary.iterrows():
        print(f"    {label:<22}: n={int(row['n_firms'])}, "
              f"mean CS-NI={row['mean_cs_ni']:>+6.2f}, "
              f"mean CS-NS={row['mean_cs_strength']:>+7.4f}, "
              f"pct_positive={row['pct_positive']:.1f}%")

    # -- Analysis 4 ----------------------------------------------------------
    print("\n[5/6] Analysis 4: Bridge firms (highest cross-sector edge fraction)")
    bridge_df = analysis_bridge_firms(df, firm_cs)

    print(f"\n  Top 15 bridge firms by cross-sector edge fraction:")
    print(f"  {'Firm':<42} {'Sector':<22} {'CS-frac':>8} {'CS-NI':>6} {'NI':>6}")
    print(f"  {'-'*90}")
    for _, r in bridge_df.head(15).iterrows():
        print(f"  {r['firm']:<42} {r['community_label']:<22} "
              f"{r['cs_edge_fraction']:>8.3f} "
              f"{int(r['net_cs_influence']):>6} "
              f"{int(r['net_influence']) if pd.notna(r['net_influence']) else 'N/A':>6}")

    # -- Analysis 5 ----------------------------------------------------------
    print("\n[6/6] Analysis 5: Issue profiles of top cross-sector directed dyads")
    dyad_df = analysis_top_dyads(df, issues)

    print(f"\n  Top 10 cross-sector dyads by net_temporal:")
    print(f"  {'Source':<30} {'Target':<30} {'Src-Sect':<22} {'Tgt-Sect':<22} "
          f"{'NT':>4} {'RBO':>7} {'CosSim':>7}")
    print(f"  {'-'*125}")
    for _, r in dyad_df.iterrows():
        print(f"  {r['source']:<30} {r['target']:<30} "
              f"{r['src_sector']:<22} {r['tgt_sector']:<22} "
              f"{int(r['net_temporal']):>4} {r['rbo_weight']:>7.4f} "
              f"{r['issue_cosine_sim']:>7.4f}")

    print(f"\n  Issue code profiles for top cross-sector dyads:")
    print(f"  {'Source → Target':<55} {'Src top-3 issues':<25} "
          f"{'Tgt top-3 issues':<25} {'Shared'}")
    print(f"  {'-'*125}")
    for _, r in dyad_df.iterrows():
        pair_label = f"{r['source']} → {r['target']}"
        print(f"  {pair_label:<55} {r['src_top3_issues']:<25} "
              f"{r['tgt_top3_issues']:<25} {r['shared_top3_issues']}")

    # -- Save outputs --------------------------------------------------------
    df.to_csv(EDGE_CSV, index=False)
    firm_cs.to_csv(FIRM_CSV, index=False)
    pair_df.to_csv(PAIR_CSV, index=False)

    print(f"\n  Edge table    → {EDGE_CSV}")
    print(f"  Firm table    → {FIRM_CSV}")
    print(f"  Pair matrix   → {PAIR_CSV}")
    print(f"  Log           → {TXT_PATH}")

    print("\n  Validation complete.")
    print("=" * 72)

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
