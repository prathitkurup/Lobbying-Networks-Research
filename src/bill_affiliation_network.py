"""
Prathit Kurup, Victoria Figueroa

Company-to-company affiliation network based on shared bill lobbying.
Edge weight = number of non-omnibus bills both companies lobbied together.

Filtering: bills lobbied by more than MAX_BILL_DF firms are excluded (see
config.py and utils/filtering.py). Omnibus bills (CARES Act, NDAA, etc.)
create spurious edges with no strategic alignment signal — analogous to
stop-word removal in NLP (Manning et al., 2008, §6.2).

Usage:
  python bill_affiliation_network.py [options]

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
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence, prevalence_summary
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import (compute_community_centralities,
                               print_community_centralities)
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "bill_affiliation_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "bill_affiliation_network.png")

LEIDEN_RESOLUTION = 1.0


# -- Edge construction --

def company_bill_edges(df, max_bill_df=MAX_BILL_DF):
    """
    For each bill, create a complete subgraph over all companies that lobbied it.
    Aggregate across bills: edge weight = total shared bill count.

    Data notes:
      • fortune500_lda_reports.csv has multiple rows per (client_name, bill_id)
        because extraction.py writes one row per bill per report. Drop duplicates
        first so only company presence matters; otherwise the inner loop generates
        a cartesian product of row counts that inflates shared-bill counts ~6×.
      • Bills lobbied by > max_bill_df firms are excluded before edge construction.
        These omnibus bills create spurious co-lobbying links with no alignment signal.
    """
    df = df.drop_duplicates(subset=["client_name", "bill_id"])

    if max_bill_df is not None:
        prevalence_summary(df)
        df = filter_bills_by_prevalence(df, max_bill_df, unit_col="bill_id")

    bill_companies = df.groupby("bill_id")["client_name"].apply(list)
    records = []
    for bill_id, companies in bill_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if companies[i] != companies[j]:
                    # Canonical ordering prevents (A,B) and (B,A) from
                    # appearing as separate edges due to non-deterministic
                    # list ordering across bills.
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
        description="Build Fortune 500 bill affiliation network."
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
    df    = load_bills_data(DATA_DIR / "fortune500_lda_reports.csv")
    edges = company_bill_edges(df)

    print(f"Bills: {df['bill_id'].nunique():,}  |  "
          f"Companies: {df['client_name'].nunique():,}")
    edge_weight_stats(edges, "shared bills (filtered)")

    G = build_graph(edges[["source", "target", "weight"]])

    # -- Leiden community detection --
    if not args.no_sweep:
        print(f"\n-- Resolution sweep (bill affiliation) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=args.resolution, seed=42
    )
    print(f"\n  Leiden (gamma={args.resolution})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="bill affiliation")

    comm_df = (
        pd.DataFrame([{"client_name": n, "community_aff": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_aff", "client_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_affiliation.csv", index=False)
    print(f"\nCommunity assignments -> {DATA_DIR / 'communities_affiliation.csv'}")

    # -- Centrality (computed before GML so attrs can be embedded) --
    # Always compute when GML is requested; also print/export if centrality_k > 0.
    cent_df = None
    if not args.no_gml or args.centrality_k > 0:
        cent_df = compute_community_centralities(G, partition)
        if args.centrality_k > 0:
            print_community_centralities(cent_df, k=args.centrality_k)
            cent_df.to_csv(DATA_DIR / "centrality_affiliation.csv", index=False)
            print(f"\nCentrality -> {DATA_DIR / 'centrality_affiliation.csv'}")

    # -- GML with enriched node attributes --
    # Includes: community, centrality measures, GA role, and k-core number.
    # Gephi reads all node attributes as importable partition/metric columns.
    if not args.no_gml:
        node_attrs = _cent_df_to_attrs(cent_df)
        kcore = nx.core_number(G)
        if node_attrs is None:
            node_attrs = {}
        node_attrs["kcore"] = kcore
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
    # --

    if args.top_k > 0:
        H = top_k_subgraph(G, k=args.top_k)
        plot_circular(H,
                      title=f"Top {args.top_k} Fortune 500 Bill Affiliation Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main(parse_args())
