"""
RBO-based issue-code similarity network (Fortune 500, 116th Congress).

Ranks each firm's issue-code portfolio by spend fraction (descending) and
computes Rank-Biased Overlap pairwise — the same methodology used for the
bill-level RBO similarity network, applied to the 76 LDA issue codes.

Design decisions vs the existing cosine issue network:
  - RBO top-weights high-spend issue codes, matching economic priority signal.
  - TOP_ISSUES=30 covers full portfolios for 97% of firms (median=6 codes).
  - p=0.85 matches the bill-level calibration (§18 in design_decisions.md).
  - No issue-code frequency filter (MAX_ISSUE_DF=None per config.py §8).

Outputs:
  data/network_edges/issue_rbo_edges.csv
  data/communities/communities_issue_rbo.csv
  data/centralities/centrality_issue_rbo.csv
  visualizations/gml/issue_rbo_similarity_network.gml
  visualizations/png/issue_rbo_similarity_network.png
"""

import sys
import itertools
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ROOT, OPENSECRETS_ISSUES_CSV
from utils.similarity import rbo_score
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "issue_rbo_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png"  / "issue_rbo_similarity_network.png")

TOP_ISSUES        = 30    # top issue codes per firm; covers full portfolios for ~97% of firms
RBO_P             = 0.85  # matches bill-level calibration (§18)
MIN_RBO           = 0.0   # no minimum threshold; keep all positive weights
TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
LEIDEN_RESOLUTION = 1.0


# -- Data preparation ---------------------------------------------------------

def build_issue_ranked_lists(issues_csv, top_issues=TOP_ISSUES):
    """Aggregate issue spend per firm, compute fracs, return top-K ranked lists."""
    df = pd.read_csv(issues_csv)

    # Aggregate across all quarters: one row per (firm, issue_code)
    agg = (
        df.groupby(["fortune_name", "issue_code"], as_index=False)["amount_allocated"]
        .sum()
        .rename(columns={"amount_allocated": "total_amount"})
    )

    # Spend fraction per issue code within each firm
    firm_totals = agg.groupby("fortune_name")["total_amount"].transform("sum")
    agg = agg[firm_totals > 0].copy()
    agg["frac"] = agg["total_amount"] / firm_totals[firm_totals > 0]

    # Build top-K ranked list per firm (by frac descending)
    ranked = {}
    for firm, grp in agg.groupby("fortune_name"):
        top = (grp.nlargest(top_issues, "frac")
                  .sort_values("frac", ascending=False)["issue_code"]
                  .tolist())
        if top:
            ranked[firm] = top

    return ranked, agg


# -- Edge construction --------------------------------------------------------

def build_issue_rbo_edges(ranked, p=RBO_P, min_rbo=MIN_RBO):
    """Compute RBO-weighted edges for all firm pairs sharing ≥1 top issue code."""
    firms   = sorted(ranked.keys())
    records = []

    for firm_a, firm_b in itertools.combinations(firms, 2):
        score = rbo_score(ranked[firm_a], ranked[firm_b], p=p)
        if score > min_rbo:
            src, tgt = (firm_a, firm_b) if firm_a < firm_b else (firm_b, firm_a)
            records.append({"source": src, "target": tgt, "weight": round(score, 6)})

    return pd.DataFrame(records) if records else pd.DataFrame(columns=["source","target","weight"])


# -- Main ---------------------------------------------------------------------

def main():
    print(f"Building issue RBO similarity network  "
          f"(TOP_ISSUES={TOP_ISSUES}, p={RBO_P})")

    ranked, agg = build_issue_ranked_lists(OPENSECRETS_ISSUES_CSV)
    print(f"  Firms with issue ranked lists: {len(ranked):,}")
    print(f"  Unique issue codes in data:    {agg['issue_code'].nunique()}")
    codes_per_firm = agg.groupby("fortune_name")["issue_code"].count()
    print(f"  Issue codes per firm — mean: {codes_per_firm.mean():.1f}  "
          f"median: {codes_per_firm.median():.0f}  max: {codes_per_firm.max()}")

    print("\nComputing pairwise RBO scores...")
    edges = build_issue_rbo_edges(ranked)
    edge_weight_stats(edges, "RBO issue similarity")

    if edges.empty:
        print("No edges — check issue data.")
        return

    edges_out = DATA_DIR / "network_edges" / "issue_rbo_edges.csv"
    edges.to_csv(edges_out, index=False)
    print(f"  Edge list -> {edges_out.name}")

    G = build_graph(edges)
    print(f"\n-- Network summary --")
    print(f"  Nodes:    {G.number_of_nodes()}")
    print(f"  Edges:    {G.number_of_edges()}")
    print(f"  Density:  {nx.density(G):.4f}")

    if RUN_SWEEP:
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="issue RBO")

    comm_df = (
        pd.DataFrame([{"firm": n, "community": cid} for n, cid in partition.items()])
        .sort_values(["community", "firm"]).reset_index(drop=True)
    )
    comm_out = DATA_DIR / "communities" / "communities_issue_rbo.csv"
    comm_df.to_csv(comm_out, index=False)
    print(f"\nCommunities -> {comm_out.name}")

    cent_df = None
    if CENTRALITY_K > 0 or WRITE_GML:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_out = DATA_DIR / "centralities" / "centrality_issue_rbo.csv"
            cent_df.to_csv(cent_out, index=False)
            print(f"\nCentrality -> {cent_out.name}")

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H, title=f"Top {TOP_K} Fortune 500 — Issue RBO Similarity Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main()
