"""
Prathit Kurup, Victoria Figueroa

Company-to-company similarity network based on bill lobbying priority rankings.

Similarity metric -- Rank-Biased Overlap (Webber et al. 2010):

  RBO(i, j, p) = (1 - p) * sum_{d=1}^{min_len} p^(d-1) * |R_i[:d] & R_j[:d]| / d

  where R_i = firm i's bills sorted by allocated spend (descending).
  p = 0.90 by default: ~65% of the score comes from agreement in the top-10
  bills, naturally emphasizing high-spend legislative priorities.

Design decision -- all-pairs (not within-community only):
  Computing RBO across all firm pairs keeps this network self-contained and
  directly comparable to the BC and cosine similarity networks. Within-community
  restriction would hard-code a dependency on bill_affiliation_network.py
  partition results and would prevent cross-community comparison.
  With ~300 firms and top-100 truncation, all-pairs is fast (~4.5M ops).

Two-stage filtering:
  Fracs (used to rank bills) are computed on ALL bills so the denominator
  (total firm budget) is preserved. Mega-bills are excluded only before
  building ranked lists, removing structurally uninformative shared entries.

Comparison note:
  All three similarity networks (bill_similarity_network, rbo_similarity_network,
  cosine_similarity_network) export CSV files to DATA_DIR with matching 'firm'
  column names so centrality tables can be joined and compared directly.

Usage:
  python rbo_similarity_network.py [options]

  --top-k K          Top-K subgraph for plot (default 20; 0 = skip plot)
  --centrality-k K   Firms printed per centrality metric (default 10; 0 = skip)
  --no-gml           Do not write a GML file
  --no-sweep         Skip the Leiden resolution sweep
  --resolution gamma Leiden resolution parameter (default 1.0)
  --rbo-p FLOAT      RBO persistence parameter (default 0.90)
  --top-bills K      Truncate ranked lists to top-K bills (default 100; 0 = all)
  --min-weight FLOAT Minimum edge weight to include (default 0.01)
"""

import argparse
import sys
import itertools
import pandas as pd
import networkx as nx

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import (aggregate_per_firm_bill, compute_zero_budget_fracs,
                               build_ranked_lists, rbo_score)
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import (compute_community_centralities,
                               print_community_centralities)
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "rbo_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "rbo_similarity_network.png")

LEIDEN_RESOLUTION = 1.0
DEFAULT_RBO_P     = 0.90
DEFAULT_TOP_BILLS = 100
DEFAULT_MIN_WEIGHT = 0.01


# -- Edge construction ---------------------------------------------------------

def company_rbo_edges(df, p=DEFAULT_RBO_P, top_bills=DEFAULT_TOP_BILLS,
                      max_bill_df=MAX_BILL_DF, min_weight=DEFAULT_MIN_WEIGHT):
    """
    Build RBO similarity edges between all firm pairs.

    Steps:
      1. Aggregate to one row per (firm, bill) by summing amounts.
      2. Compute fracs (portfolio shares) for the full bill universe.
      3. Apply prevalence filter to exclude mega-bills before ranking.
         These bills are lobbied by nearly every firm, so their position
         in a ranked list carries no discriminative signal.
      4. Build per-firm ranked bill lists (sorted by amount descending).
      5. Truncate lists to top_bills (focuses RBO on actual priorities).
      6. Compute RBO for all firm pairs; keep edges >= min_weight.

    Returns DataFrame with columns [source, target, weight].
    """
    df_agg = aggregate_per_firm_bill(df)
    df_agg = compute_zero_budget_fracs(df_agg)  # adds frac, drops zero-budget firms

    # Apply prevalence filter ONLY for building ranked lists (same two-stage
    # logic as bill_similarity_network). Fracs are already computed on all bills.
    df_for_ranking = df_agg
    if max_bill_df is not None:
        df_for_ranking = filter_bills_by_prevalence(
            df_agg, max_bill_df, unit_col="bill_id"
        )

    ranked = build_ranked_lists(df_for_ranking, top_bills=top_bills)
    firms  = sorted(ranked.keys())

    print(f"  Firms with ranked lists: {len(firms):,}")
    list_lens = [len(ranked[f]) for f in firms]
    print(f"  Bills per list  -- mean: {sum(list_lens)/len(list_lens):.1f}  "
          f"median: {sorted(list_lens)[len(list_lens)//2]}  "
          f"max: {max(list_lens)}")

    records = []
    for f1, f2 in itertools.combinations(firms, 2):
        w = rbo_score(ranked[f1], ranked[f2], p=p)
        if w >= min_weight:
            src, tgt = (f1, f2) if f1 < f2 else (f2, f1)
            records.append({"source": src, "target": tgt, "weight": w})

    if not records:
        return pd.DataFrame(columns=["source", "target", "weight"])

    edges = pd.DataFrame(records)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    return edges[edges["weight"] > 0]


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Fortune 500 RBO bill-priority similarity network."
    )
    p.add_argument("--top-k",        type=int,   default=20,
                   help="Nodes in the top-k subgraph/plot (0 = skip plot).")
    p.add_argument("--centrality-k", type=int,   default=10,
                   help="Firms printed per centrality metric (0 = skip).")
    p.add_argument("--no-gml",       action="store_true",
                   help="Do not write the GML file.")
    p.add_argument("--no-sweep",     action="store_true",
                   help="Skip the Leiden resolution sweep.")
    p.add_argument("--resolution",   type=float, default=LEIDEN_RESOLUTION,
                   help=f"Leiden resolution gamma (default {LEIDEN_RESOLUTION}).")
    p.add_argument("--rbo-p",        type=float, default=DEFAULT_RBO_P,
                   help=f"RBO persistence parameter (default {DEFAULT_RBO_P}). "
                        "Higher = slower decay, more weight on lower-ranked bills.")
    p.add_argument("--top-bills",    type=int,   default=DEFAULT_TOP_BILLS,
                   help=f"Truncate ranked lists to top K bills (default {DEFAULT_TOP_BILLS}; "
                        "0 = no truncation).")
    p.add_argument("--min-weight",   type=float, default=DEFAULT_MIN_WEIGHT,
                   help=f"Minimum RBO edge weight to include (default {DEFAULT_MIN_WEIGHT}).")
    return p.parse_args()


# -- Main ----------------------------------------------------------------------

def main(args):
    df_raw = load_bills_data(DATA_DIR / "fortune500_lda_reports.csv")

    print(f"Bills: {df_raw['bill_id'].nunique():,}  |  "
          f"Companies: {df_raw['client_name'].nunique():,}")
    print(f"RBO params: p={args.rbo_p}  top_bills={args.top_bills}  "
          f"min_weight={args.min_weight}")

    edges = company_rbo_edges(
        df_raw,
        p=args.rbo_p,
        top_bills=args.top_bills,
        min_weight=args.min_weight,
    )
    edge_weight_stats(edges, f"RBO similarity (p={args.rbo_p})")
    edges.to_csv(DATA_DIR / "rbo_edges.csv", index=False)
    print(f"  RBO edges -> {DATA_DIR / 'rbo_edges.csv'}")

    G = build_graph(edges)

    # -- Leiden community detection --------------------------------------------
    if not args.no_sweep:
        print("\n-- Resolution sweep (RBO similarity) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=args.resolution, seed=42
    )
    print(f"\n  Leiden (gamma={args.resolution})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="RBO similarity")

    comm_df = (
        pd.DataFrame([{"client_name": n, "community_rbo": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_rbo", "client_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_rbo.csv", index=False)
    print(f"\n  Community assignments -> {DATA_DIR / 'communities_rbo.csv'}")

    # -- Centrality (computed before GML so attrs can be embedded) -------------
    cent_df = None
    if not args.no_gml or args.centrality_k > 0:
        cent_df = compute_community_centralities(G, partition)
        if args.centrality_k > 0:
            print_community_centralities(cent_df, k=args.centrality_k)
            cent_df.to_csv(DATA_DIR / "centrality_rbo.csv", index=False)
            print(f"\n  Centrality -> {DATA_DIR / 'centrality_rbo.csv'}")

    # -- GML with enriched node attributes ------------------------------------
    if not args.no_gml:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
    # -------------------------------------------------------------------------

    if args.top_k > 0:
        H = top_k_subgraph(G, k=args.top_k)
        plot_circular(H,
                      title=f"Top {args.top_k} Fortune 500 RBO Similarity Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main(parse_args())
