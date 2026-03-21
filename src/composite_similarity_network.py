"""
Composite company-to-company lobbying similarity network.

Outputs: composite_edges.csv, communities_composite.csv,
centrality_composite.csv, and GML visualization.
"""

import sys

import pandas as pd

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT
from utils.network_building import (
    build_graph_with_attrs, write_gml_with_communities,
    _cent_df_to_attrs, top_k_subgraph, edge_weight_stats,
)
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "composite_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "composite_similarity_network.png")

TOP_K           = 20
CENTRALITY_K    = 10
WRITE_GML       = True
RUN_SWEEP       = False
LEIDEN_RESOLUTION = 0.25


def build_composite_edges():
    """Load and composite three component edge CSVs."""
    affil_path  = DATA_DIR / "affiliation_edges.csv"
    cosine_path = DATA_DIR / "cosine_edges.csv"
    rbo_path    = DATA_DIR / "rbo_edges.csv"

    for p in (affil_path, cosine_path, rbo_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found. "
                "Run bill_affiliation_network.py, cosine_similarity_network.py, "
                "and rbo_similarity_network.py first."
            )

    affil  = pd.read_csv(affil_path).rename(columns={"weight": "shared_n"})
    cosine = pd.read_csv(cosine_path).rename(columns={"weight": "cosine_weight"})
    rbo    = pd.read_csv(rbo_path).rename(columns={"weight": "rbo_weight"})

    # Canonical ordering: smaller firm name always in 'source'
    for df in (affil, cosine, rbo):
        mask = df["source"] > df["target"]
        df.loc[mask, ["source", "target"]] = df.loc[mask, ["target", "source"]].values

    # Inner join: composite edge only when all three signals are present
    merged = (
        affil
        .merge(cosine, on=["source", "target"], how="inner")
        .merge(rbo,    on=["source", "target"], how="inner")
    )

    merged["weight"] = merged["affil_norm"] * merged["cosine_weight"] * merged["rbo_weight"] * 1000
    merged = merged[merged["weight"] > 0].reset_index(drop=True)

    print(f"Composite edges: {len(merged):,}  "
          f"(weight range {merged['weight'].min():.6f} - {merged['weight'].max():.6f})")
    print(f"  Unique firms: {len(set(merged['source']) | set(merged['target']))}")

    return merged[[
        "source", "target", "weight",
        "affil_norm", "shared_n", "cosine_weight", "rbo_weight"
    ]]


def main():
    print("=" * 60)
    print("COMPOSITE SIMILARITY NETWORK  (affil_norm x cosine x rbo)")
    print("=" * 60)

    edges = build_composite_edges()

    out_path = DATA_DIR / "composite_edges.csv"
    edges.to_csv(out_path, index=False)
    print(f"Edges -> {out_path}")

    G = build_graph_with_attrs(edges, weight_col="weight")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    edge_weight_stats(edges, "composite (affil_norm x cosine x rbo)")

    if RUN_SWEEP:
        sweep_resolution(G, weight_attr="weight",
                         resolutions=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

    partition, Q, comm_summary = detect_communities(
        G, weight_attr="weight", resolution=LEIDEN_RESOLUTION, seed=42
    )
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="composite")

    comm_df = pd.DataFrame(
        [(firm, cid) for firm, cid in partition.items()],
        columns=["firm", "community"]
    )
    comm_path = DATA_DIR / "communities_composite.csv"
    comm_df.to_csv(comm_path, index=False)
    print(f"\nCommunities -> {comm_path}")

    cent_df = compute_community_centralities(G, partition, weight_attr="weight")
    print_community_centralities(cent_df, k=CENTRALITY_K)

    cent_path = DATA_DIR / "centrality_composite.csv"
    cent_df.to_csv(cent_path, index=False)
    print(f"\nCentrality -> {cent_path}")

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
        print(f"\nGML -> {GML_PATH}")

    if TOP_K > 0:
        sub = top_k_subgraph(G, k=TOP_K)
        plot_circular(sub, title="Composite Similarity Network (top-K)", path=PNG_PATH)
        print(f"\nPlot -> {PNG_PATH}")


if __name__ == "__main__":
    main()
