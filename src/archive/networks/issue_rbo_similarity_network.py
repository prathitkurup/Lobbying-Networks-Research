"""
[ARCHIVED] Supporting network: RBO-based issue-code similarity between Fortune 500 firms.

Applies RBO to 76 LDA issue-code portfolios (same methodology as bill-level RBO).
Outputs (archived): data/archive/network_edges/issue_rbo_edges.csv,
data/archive/communities/communities_issue_rbo.csv,
data/archive/centralities/centrality_issue_rbo.csv
"""

import sys
import itertools
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, ROOT, OPENSECRETS_ISSUES_CSV
from utils.similarity import rbo_score
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

ARCHIVE           = DATA_DIR / "archive"
GML_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "issue_rbo_similarity_network.gml")
PNG_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "issue_rbo_similarity_network.png")

TOP_ISSUES        = 30
RBO_P             = 0.85
MIN_RBO           = 0.0
TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
LEIDEN_RESOLUTION = 1.0


def build_issue_ranked_lists(issues_csv, top_issues=TOP_ISSUES):
    """Aggregate issue spend per firm and return top-K ranked issue-code lists."""
    df = pd.read_csv(issues_csv)
    agg = (df.groupby(["fortune_name", "issue_code"], as_index=False)["amount_allocated"]
             .sum().rename(columns={"amount_allocated": "total_amount"}))
    firm_totals = agg.groupby("fortune_name")["total_amount"].transform("sum")
    agg = agg[firm_totals > 0].copy()
    agg["frac"] = agg["total_amount"] / firm_totals[firm_totals > 0]
    ranked = {firm: (grp.nlargest(top_issues, "frac")
                        .sort_values("frac", ascending=False)["issue_code"]
                        .tolist())
              for firm, grp in agg.groupby("fortune_name")}
    return {f: lst for f, lst in ranked.items() if lst}, agg


def build_issue_rbo_edges(ranked, p=RBO_P, min_rbo=MIN_RBO):
    """Compute RBO-weighted edges for all firm pairs sharing ≥1 top issue code."""
    records = []
    for firm_a, firm_b in itertools.combinations(sorted(ranked), 2):
        score = rbo_score(ranked[firm_a], ranked[firm_b], p=p)
        if score > min_rbo:
            src, tgt = (firm_a, firm_b) if firm_a < firm_b else (firm_b, firm_a)
            records.append({"source": src, "target": tgt, "weight": round(score, 6)})
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["source", "target", "weight"])


def main():
    ranked, agg = build_issue_ranked_lists(OPENSECRETS_ISSUES_CSV)
    print(f"Firms with issue ranked lists: {len(ranked):,}  |  "
          f"Unique issue codes: {agg['issue_code'].nunique()}")
    edges = build_issue_rbo_edges(ranked)
    edge_weight_stats(edges, "RBO issue similarity")
    edges.to_csv(ARCHIVE / "network_edges" / "issue_rbo_edges.csv", index=False)

    G = build_graph(edges)
    if RUN_SWEEP:
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="issue RBO")

    comm_df = (
        pd.DataFrame([{"firm": n, "community": cid} for n, cid in partition.items()])
        .sort_values(["community", "firm"]).reset_index(drop=True)
    )
    comm_df.to_csv(ARCHIVE / "communities" / "communities_issue_rbo.csv", index=False)

    cent_df = None
    if CENTRALITY_K > 0 or WRITE_GML:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_df.to_csv(ARCHIVE / "centralities" / "centrality_issue_rbo.csv", index=False)

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H, title=f"Top {TOP_K} Fortune 500 Issue RBO Similarity Network", path=PNG_PATH)


if __name__ == "__main__":
    main()
