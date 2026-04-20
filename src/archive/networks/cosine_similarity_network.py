"""
[ARCHIVED] Supporting network: cosine similarity of bill-portfolio budget vectors.

Outputs (archived): data/archive/network_edges/cosine_edges.csv,
data/archive/communities/communities_cosine.csv,
data/archive/centralities/centrality_cosine.csv
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import aggregate_per_firm_bill, compute_zero_budget_fracs, build_frac_matrix
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

ARCHIVE           = DATA_DIR / "archive"
GML_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "cosine_similarity_network.gml")
PNG_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "cosine_similarity_network.png")

TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
MIN_WEIGHT        = 0.0
LEIDEN_RESOLUTION = 0.6


def company_cosine_edges(df, max_bill_df=MAX_BILL_DF, min_weight=MIN_WEIGHT):
    """Build cosine similarity edges between all firm pairs."""
    df_agg = aggregate_per_firm_bill(df)
    df_agg = compute_zero_budget_fracs(df_agg)
    df_mat = (filter_bills_by_prevalence(df_agg, max_bill_df, unit_col="bill_number")
              if max_bill_df is not None else df_agg)
    pivot, firms, bills = build_frac_matrix(df_mat)
    print(f"  Frac matrix: {len(firms):,} firms x {len(bills):,} bills  "
          f"(sparsity: {100*(pivot.values == 0).mean():.1f}% zeros)")
    mat = pivot.values.astype(np.float64)
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
    df_raw = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    print(f"Bills: {df_raw['bill_number'].nunique():,}  |  Companies: {df_raw['fortune_name'].nunique():,}")
    edges = company_cosine_edges(df_raw)
    edge_weight_stats(edges, "cosine similarity (budget vectors)")
    edges.to_csv(ARCHIVE / "network_edges" / "cosine_edges.csv", index=False)

    G = build_graph(edges)
    if RUN_SWEEP:
        sweep_resolution(G, resolutions=[0.4, 0.45, 0.5, 0.55, 0.6, 0.7], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="cosine similarity")

    comm_df = (
        pd.DataFrame([{"fortune_name": n, "community_cosine": cid} for n, cid in partition.items()])
        .sort_values(["community_cosine", "fortune_name"]).reset_index(drop=True)
    )
    comm_df.to_csv(ARCHIVE / "communities" / "communities_cosine.csv", index=False)

    cent_df = None
    if WRITE_GML or CENTRALITY_K > 0:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_df.to_csv(ARCHIVE / "centralities" / "centrality_cosine.csv", index=False)

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H, title=f"Top {TOP_K} Fortune 500 Cosine Similarity Network", path=PNG_PATH)


if __name__ == "__main__":
    main()
