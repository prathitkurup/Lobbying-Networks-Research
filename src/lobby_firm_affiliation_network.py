"""
Company-to-company affiliation network based on shared lobbying firms (registrants).

Outputs: lobby_firm_affiliation_edges.csv, communities_lobby_firm.csv,
centrality_lobby_firm.csv, and GML visualization.
"""

import sys
import pandas as pd
import networkx as nx

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT
from utils.data_loading import load_lobby_firm_data
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import (compute_community_centralities,
                               print_community_centralities)
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "lobby_firm_affiliation_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "lobby_firm_affiliation_network.png")

TOP_K           = 20
CENTRALITY_K    = 10
WRITE_GML       = True
RUN_SWEEP       = False
LEIDEN_RESOLUTION = 1.0


def company_registrant_edges(df):
    """Build shared-registrant edge list."""
    registrant_companies = (
        df.groupby("registrant")["fortune_name"]
          .apply(lambda x: list(set(x)))
    )

    records = []
    for reg_id, companies in registrant_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if companies[i] != companies[j]:
                    src, tgt = ((companies[i], companies[j])
                                if companies[i] < companies[j]
                                else (companies[j], companies[i]))
                    records.append({"source": src, "target": tgt})

    if not records:
        return pd.DataFrame(columns=["source", "target", "weight"])
    edges = (pd.DataFrame(records)
               .groupby(["source", "target"])
               .size()
               .reset_index(name="weight"))
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    return edges[edges["weight"] > 0]


def main():
    df = load_lobby_firm_data(DATA_DIR / "opensecrets_lda_reports.csv")

    print(f"Registrants: {df['registrant'].nunique():,}  |  "
          f"Companies: {df['fortune_name'].nunique():,}")

    edges = company_registrant_edges(df)
    edge_weight_stats(edges, "shared registrants")

    G = build_graph(edges)

    if RUN_SWEEP:
        print(f"\n-- Resolution sweep (lobby firm affiliation) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=LEIDEN_RESOLUTION, seed=42
    )
    print(f"\n  Leiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="lobby firm affiliation")

    comm_df = (
        pd.DataFrame([{"fortune_name": n, "community_lobby": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_lobby", "fortune_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_lobby_firm.csv", index=False)
    print(f"  Community assignments -> {DATA_DIR / 'communities_lobby_firm.csv'}")

    cent_df = None
    if WRITE_GML or CENTRALITY_K > 0:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_df.to_csv(DATA_DIR / "centrality_lobby_firm.csv", index=False)
            print(f"  Centrality -> {DATA_DIR / 'centrality_lobby_firm.csv'}")

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H,
                      title=f"Top {TOP_K} Fortune 500 Lobby Firm Affiliation Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main()
