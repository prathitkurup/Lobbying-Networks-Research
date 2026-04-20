"""
[ARCHIVED] Supporting network: cosine similarity of issue-code portfolios.

Outputs (archived): data/archive/network_edges/issue_edges.csv,
data/archive/communities/communities_issue.csv,
data/archive/centralities/centrality_issue.csv
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_ISSUE_DF
from utils.data_loading import load_issues_data
from utils.filtering import filter_bills_by_prevalence
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

ARCHIVE           = DATA_DIR / "archive"
GML_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "issue_similarity_network.gml")
PNG_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "issue_similarity_network.png")

TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
MIN_WEIGHT        = 0.0
LEIDEN_RESOLUTION = 1.0


def company_issue_edges(df, max_issue_df=MAX_ISSUE_DF, min_weight=MIN_WEIGHT):
    """Build cosine similarity edges for issue-code portfolios."""
    df = df.groupby(["fortune_name", "issue_code"], as_index=False)["amount_allocated"].sum()
    company_totals = df.groupby("fortune_name")["amount_allocated"].sum().rename("total_budget")
    df = df.merge(company_totals, on="fortune_name", how="left")
    zero_budget = company_totals[company_totals == 0].index.tolist()
    if zero_budget:
        print(f"  Warning: {len(zero_budget)} firm(s) excluded (zero budget): {zero_budget}")
        df = df[df["total_budget"] > 0].copy()
    df["frac"] = df["amount_allocated"] / df["total_budget"]
    df_mat = (filter_bills_by_prevalence(df, max_issue_df, unit_col="issue_code")
              if max_issue_df is not None else df)
    pivot = df_mat.pivot_table(index="fortune_name", columns="issue_code",
                                values="frac", fill_value=0.0)
    firms = list(pivot.index)
    mat   = pivot.values.astype(np.float64)
    print(f"  Frac matrix: {len(firms):,} firms x {pivot.shape[1]} issue codes  "
          f"(sparsity: {100*(mat == 0).mean():.1f}% zeros)")
    sim = cosine_similarity(mat)
    n = len(firms)
    rows, cols = np.triu_indices(n, k=1)
    records = []
    for i, j in zip(rows, cols):
        w = float(sim[i, j])
        if w >= min_weight:
            f1, f2 = firms[i], firms[j]
            src, tgt = (f1, f2) if f1 < f2 else (f2, f1)
            records.append({"source": src, "target": tgt, "weight": round(w, 6)})
    if not records:
        return pd.DataFrame(columns=["source", "target", "weight"])
    edges = pd.DataFrame(records)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    return edges[edges["weight"] > 0]


def main():
    df    = load_issues_data(DATA_DIR / "opensecrets_lda_issues.csv")
    edges = company_issue_edges(df)
    print(f"Issues: {df['issue_code'].nunique()}  |  Companies: {df['fortune_name'].nunique():,}")
    edge_weight_stats(edges, "cosine issue similarity")
    edges.to_csv(ARCHIVE / "network_edges" / "issue_edges.csv", index=False)

    G = build_graph(edges)
    if RUN_SWEEP:
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="issue similarity")

    comm_df = (
        pd.DataFrame([{"fortune_name": n, "community_issue": cid} for n, cid in partition.items()])
        .sort_values(["community_issue", "fortune_name"]).reset_index(drop=True)
    )
    comm_df.to_csv(ARCHIVE / "communities" / "communities_issue.csv", index=False)

    cent_df = None
    if WRITE_GML or CENTRALITY_K > 0:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_df.to_csv(ARCHIVE / "centralities" / "centrality_issue.csv", index=False)

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H, title=f"Top {TOP_K} Fortune 500 Issue Similarity Network", path=PNG_PATH)


if __name__ == "__main__":
    main()
