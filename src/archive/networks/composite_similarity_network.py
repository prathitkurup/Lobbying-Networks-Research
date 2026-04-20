"""
[ARCHIVED] Supporting network: composite similarity (affil_norm × cosine × rbo).

Requires affiliation_edges.csv, cosine_edges.csv, rbo_edges.csv in data/archive/network_edges/.
Outputs (archived): data/archive/network_edges/composite_edges.csv,
data/archive/communities/communities_composite.csv,
data/archive/centralities/centrality_composite.csv
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT
from utils.network_building import (build_graph_with_attrs, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

ARCHIVE           = DATA_DIR / "archive"
GML_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "composite_similarity_network.gml")
PNG_PATH          = str(ROOT / "visualizations" / "archive" / "undirected" / "composite_similarity_network.png")

TOP_K             = 20
CENTRALITY_K      = 10
WRITE_GML         = True
RUN_SWEEP         = False
LEIDEN_RESOLUTION = 0.25


def build_composite_edges():
    """Inner-join affiliation, cosine, and RBO edges; weight = affil_norm × cosine × rbo × 1000."""
    net_dir = ARCHIVE / "network_edges"
    affil  = pd.read_csv(net_dir / "affiliation_edges.csv").rename(columns={"weight": "shared_n"})
    cosine = pd.read_csv(net_dir / "cosine_edges.csv").rename(columns={"weight": "cosine_weight"})
    rbo    = pd.read_csv(net_dir / "rbo_edges.csv").rename(columns={"weight": "rbo_weight"})
    for df in (affil, cosine, rbo):
        mask = df["source"] > df["target"]
        df.loc[mask, ["source", "target"]] = df.loc[mask, ["target", "source"]].values
    merged = (affil
              .merge(cosine, on=["source", "target"], how="inner")
              .merge(rbo,    on=["source", "target"], how="inner"))
    merged["weight"] = merged["affil_norm"] * merged["cosine_weight"] * merged["rbo_weight"] * 1000
    return merged[merged["weight"] > 0][
        ["source", "target", "weight", "affil_norm", "shared_n", "cosine_weight", "rbo_weight"]
    ].reset_index(drop=True)


def main():
    edges = build_composite_edges()
    edge_weight_stats(edges, "composite (affil_norm × cosine × rbo)")
    edges.to_csv(ARCHIVE / "network_edges" / "composite_edges.csv", index=False)

    G = build_graph_with_attrs(edges, weight_col="weight")
    if RUN_SWEEP:
        sweep_resolution(G, weight_attr="weight",
                         resolutions=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

    partition, Q, comm_summary = detect_communities(G, weight_attr="weight",
                                                    resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="composite")

    comm_df = pd.DataFrame([(firm, cid) for firm, cid in partition.items()],
                           columns=["firm", "community"])
    comm_df.to_csv(ARCHIVE / "communities" / "communities_composite.csv", index=False)

    cent_df = compute_community_centralities(G, partition, weight_attr="weight")
    if CENTRALITY_K > 0:
        print_community_centralities(cent_df, k=CENTRALITY_K)
        cent_df.to_csv(ARCHIVE / "centralities" / "centrality_composite.csv", index=False)

    if WRITE_GML:
        write_gml_with_communities(G, partition, GML_PATH, _cent_df_to_attrs(cent_df))

    if TOP_K > 0:
        sub = top_k_subgraph(G, k=TOP_K)
        plot_circular(sub, title="Composite Similarity Network (top-K)", path=PNG_PATH)


if __name__ == "__main__":
    main()
