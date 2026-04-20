"""
[ARCHIVED] Supporting network: shared human-lobbyist affiliation between Fortune 500 firms.

Outputs (archived): data/network_edges/lobbyist_affiliation_edges.csv (kept in active
network_edges/ because affiliation_mediated_adoption.py reads it),
data/archive/communities/communities_lobbyist_affiliation.csv,
data/archive/centralities/centrality_lobbyist_affiliation.csv
"""

import sys
from pathlib import Path
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, OPENSECRETS_OUTPUT_CSV, ROOT
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

ARCHIVE           = DATA_DIR / "archive"
GML_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "lobbyist_affiliation_network.gml")
PNG_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "lobbyist_affiliation_network.png")

TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
LEIDEN_RESOLUTION = 1.0


def build_pairs_from_reports(df):
    """Derive unique (lobbyist_name, fortune_name) pairs from pipe-separated lobbyists column."""
    df = df.dropna(subset=["lobbyists", "fortune_name"])
    pairs = (
        df[["fortune_name", "lobbyists"]]
        .assign(lobbyist_name=df["lobbyists"].str.split("|"))
        .explode("lobbyist_name")
    )
    pairs["lobbyist_name"] = pairs["lobbyist_name"].str.strip()
    pairs = pairs[pairs["lobbyist_name"] != ""]
    return pairs[["lobbyist_name", "fortune_name"]].drop_duplicates().reset_index(drop=True)


def build_lobbyist_affiliation_edges(pairs_df):
    """Build edge list where weight is number of shared lobbyists."""
    lobbyist_clients = pairs_df.groupby("lobbyist_name")["fortune_name"].apply(lambda x: sorted(set(x)))
    records = []
    for _, clients in lobbyist_clients.items():
        n = len(clients)
        for i in range(n):
            for j in range(i + 1, n):
                records.append({"source": clients[i], "target": clients[j]})
    if not records:
        return pd.DataFrame(columns=["source", "target", "weight"])
    edges = (pd.DataFrame(records)
               .groupby(["source", "target"])
               .size()
               .reset_index(name="weight"))
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    return edges[edges["weight"] > 0].reset_index(drop=True)


def main():
    reports = pd.read_csv(OPENSECRETS_OUTPUT_CSV)
    pairs   = build_pairs_from_reports(reports)
    print(f"Unique lobbyist-client pairs: {len(pairs):,}  |  Firms: {pairs['fortune_name'].nunique()}")

    edges = build_lobbyist_affiliation_edges(pairs)
    edge_weight_stats(edges, "shared lobbyists")

    # lobbyist_affiliation_edges.csv stays in network_edges/ (read by affiliation_mediated_adoption.py)
    edges.to_csv(DATA_DIR / "network_edges" / "lobbyist_affiliation_edges.csv", index=False)

    G = build_graph(edges)
    if RUN_SWEEP:
        sweep_resolution(G, resolutions=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="lobbyist affiliation")

    comm_df = (
        pd.DataFrame([{"firm": n, "community": cid} for n, cid in partition.items()])
        .sort_values(["community", "firm"]).reset_index(drop=True)
    )
    comm_df.to_csv(ARCHIVE / "communities" / "communities_lobbyist_affiliation.csv", index=False)

    cent_df = None
    if CENTRALITY_K > 0 or WRITE_GML:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_df.to_csv(ARCHIVE / "centralities" / "centrality_lobbyist_affiliation.csv", index=False)

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H, title=f"Top {TOP_K} Fortune 500 Lobbyist Affiliation Network", path=PNG_PATH)


if __name__ == "__main__":
    main()
