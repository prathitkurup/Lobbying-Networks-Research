"""
Layer 2 - Lobbyist Affiliation Network (Fortune 500, 116th Congress).

Edges: number of unique human lobbyists retained by both companies.
Outputs: lobbyist_affiliation_edges.csv, communities_lobbyist_affiliation.csv,
centrality_lobbyist_affiliation.csv, and GML visualization.

Requires: opensecrets_lda_reports.csv from opensecrets_extraction.py.
Lobbyist names are parsed inline from the pipe-separated `lobbyists` column.
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, OPENSECRETS_OUTPUT_CSV, ROOT
from utils.network_building import (
    build_graph,
    write_gml_with_communities,
    _cent_df_to_attrs,
    top_k_subgraph,
    edge_weight_stats,
)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GML_PATH = str(ROOT / "visualizations" / "gml" / "lobbyist_affiliation_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "lobbyist_affiliation_network.png")

TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
LEIDEN_RESOLUTION = 1.0


def build_pairs_from_reports(df):
    """
    Derive unique (lobbyist_name, fortune_name) pairs from the pipe-separated
    `lobbyists` column in opensecrets_lda_reports.csv.

    Returns a DataFrame with columns [lobbyist_name, fortune_name] where each
    row is a unique lobbyist-client pairing across all reports.
    """
    df = df.dropna(subset=["lobbyists", "fortune_name"])
    # Explode pipe-separated lobbyist names into one row per lobbyist
    pairs = (
        df[["fortune_name", "lobbyists"]]
        .assign(lobbyist_name=df["lobbyists"].str.split("|"))
        .explode("lobbyist_name")
    )
    pairs["lobbyist_name"] = pairs["lobbyist_name"].str.strip()
    pairs = pairs[pairs["lobbyist_name"] != ""]
    # One unique (lobbyist, client) pair regardless of how many reports mention them
    return pairs[["lobbyist_name", "fortune_name"]].drop_duplicates().reset_index(drop=True)


def build_lobbyist_affiliation_edges(pairs_df):
    """Build edge list where weight is number of shared lobbyists."""
    lobbyist_clients = (
        pairs_df
        .groupby("lobbyist_name")["fortune_name"]
        .apply(lambda x: sorted(set(x)))
    )

    records = []
    for _, clients in lobbyist_clients.items():
        n = len(clients)
        for i in range(n):
            for j in range(i + 1, n):
                records.append({"source": clients[i], "target": clients[j]})

    if not records:
        return pd.DataFrame(columns=["source", "target", "weight"])

    edges = (
        pd.DataFrame(records)
        .groupby(["source", "target"])
        .size()
        .reset_index(name="weight")
    )
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    return edges[edges["weight"] > 0].reset_index(drop=True)


def main():
    reports = pd.read_csv(OPENSECRETS_OUTPUT_CSV)
    print(f"Loaded {OPENSECRETS_OUTPUT_CSV.name}")
    print(f"  Reports rows                 : {len(reports):,}")
    print(f"  Fortune 500 firms covered    : {reports['fortune_name'].nunique()}")

    pairs = build_pairs_from_reports(reports)
    print(f"  Unique lobbyist-client pairs : {len(pairs):,}")
    print(f"  Unique lobbyist names        : {pairs['lobbyist_name'].nunique():,}")
    print(f"  Fortune 500 firms covered    : {pairs['fortune_name'].nunique()}")

    multi = pairs.groupby("lobbyist_name")["fortune_name"].nunique()
    n_multi = int((multi >= 2).sum())
    print(f"  Lobbyists spanning >= 2 firms: {n_multi:,}  "
          f"({100 * n_multi / max(len(multi), 1):.1f}% of unique lobbyists)")

    # -- Build edges ------------------------------------------------------
    print("\nBuilding lobbyist affiliation edges...")
    edges = build_lobbyist_affiliation_edges(pairs)
    edge_weight_stats(edges, "shared lobbyists")

    if edges.empty:
        print("No edges found — check opensecrets_extraction.py output.")
        return

    edges_out = DATA_DIR / "lobbyist_affiliation_edges.csv"
    edges.to_csv(edges_out, index=False)
    print(f"  Edge list -> {edges_out}")

    # -- Build NetworkX graph --------------------------------------------
    G = build_graph(edges)
    print(f"\n-- Network summary --")
    print(f"  Nodes     : {G.number_of_nodes()}")
    print(f"  Edges     : {G.number_of_edges()}")
    print(f"  Density   : {nx.density(G):.4f}")
    print(f"  Components: {nx.number_connected_components(G)}")

    if RUN_SWEEP:
        print(f"\n-- Resolution sweep (lobbyist affiliation) --")
        sweep_resolution(G, resolutions=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=LEIDEN_RESOLUTION, seed=42
    )
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="lobbyist affiliation")

    comm_df = (
        pd.DataFrame([{"firm": n, "community": cid}
                      for n, cid in partition.items()])
        .sort_values(["community", "firm"])
        .reset_index(drop=True)
    )
    comm_out = DATA_DIR / "communities_lobbyist_affiliation.csv"
    comm_df.to_csv(comm_out, index=False)
    print(f"\nCommunities --> {comm_out}")

    cent_df = None
    if CENTRALITY_K > 0 or WRITE_GML:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_out = DATA_DIR / "centrality_lobbyist_affiliation.csv"
            cent_df.to_csv(cent_out, index=False)
            print(f"\nCentrality -> {cent_out}")

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(
            H,
            title=f"Top {TOP_K} Fortune 500 - Lobbyist Affiliation Network (Layer 2)",
            path=PNG_PATH,
        )


if __name__ == "__main__":
    main()
