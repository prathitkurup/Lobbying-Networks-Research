"""
Prathit Kurup, Victoria Figueroa

Company-to-company affiliation network based on shared lobbying firms (registrants).
Edge weight = number of unique registrants retained by both companies.

Self-filers (companies lobbying for themselves without an external firm) are
excluded; shared self-filer registrant IDs carry no information about
organizational ties through external lobbying firms.

Usage:
  python lobby_firm_affiliation_network.py [options]

  --top-k K          Top-K subgraph for plot (default 20; 0 = skip plot)
  --centrality-k K   Firms printed per centrality metric (default 10; 0 = skip)
  --no-gml           Do not write a GML file
  --no-sweep         Skip the Leiden resolution sweep
  --resolution γ     Leiden resolution parameter (default 1.0)
"""

import argparse
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

LEIDEN_RESOLUTION = 1.0


# -- Edge construction --

def company_registrant_edges(df):
    """
    For each registrant, create a complete subgraph over all companies that
    retained it. Aggregate across registrants: edge weight = shared registrant count.

    Data note:
      list(set(x)) deduplicates companies per registrant (a firm can appear
      multiple times in a registrant's filings across reports). Canonical pair
      ordering is required because set() iteration is non-deterministic, so
      (A,B) and (B,A) can appear from different registrants and be stored as
      separate rows in the groupby without it.
    """
    registrant_companies = (
        df.groupby("registrant_id")["client_name"]
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


# -- CLI --

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Fortune 500 lobby-firm affiliation network."
    )
    p.add_argument("--top-k",        type=int,   default=20,
                   help="Nodes in the top-k subgraph/plot (0 = skip plot).")
    p.add_argument("--centrality-k", type=int,   default=10,
                   help="Firms printed per centrality metric (0 = skip all centrality).")
    p.add_argument("--no-gml",       action="store_true",
                   help="Do not write the GML file.")
    p.add_argument("--no-sweep",     action="store_true",
                   help="Skip the Leiden resolution sweep.")
    p.add_argument("--resolution",   type=float, default=LEIDEN_RESOLUTION,
                   help=f"Leiden resolution gamma (default {LEIDEN_RESOLUTION}).")
    return p.parse_args()


# -- Main --

def main(args):
    df = load_lobby_firm_data(DATA_DIR / "fortune500_lda_reports.csv")

    print(f"Registrants: {df['registrant_id'].nunique():,}  |  "
          f"Companies: {df['client_name'].nunique():,}")

    edges = company_registrant_edges(df)
    edge_weight_stats(edges, "shared registrants")

    G = build_graph(edges)

    # -- Leiden community detection --
    if not args.no_sweep:
        print(f"\n-- Resolution sweep (lobby firm affiliation) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=args.resolution, seed=42
    )
    print(f"\n  Leiden (gamma={args.resolution})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="lobby firm affiliation")

    comm_df = (
        pd.DataFrame([{"client_name": n, "community_lobby": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_lobby", "client_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_lobby_firm.csv", index=False)
    print(f"  Community assignments -> {DATA_DIR / 'communities_lobby_firm.csv'}")

    # -- Centrality (computed before GML so attrs can be embedded) --
    cent_df = None
    if not args.no_gml or args.centrality_k > 0:
        cent_df = compute_community_centralities(G, partition)
        if args.centrality_k > 0:
            print_community_centralities(cent_df, k=args.centrality_k)
            cent_df.to_csv(DATA_DIR / "centrality_lobby_firm.csv", index=False)
            print(f"  Centrality -> {DATA_DIR / 'centrality_lobby_firm.csv'}")

    # -- GML with enriched node attributes --
    if not args.no_gml:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
    # --

    if args.top_k > 0:
        H = top_k_subgraph(G, k=args.top_k)
        plot_circular(H,
                      title=f"Top {args.top_k} Fortune 500 Lobby Firm Affiliation Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main(parse_args())
