"""
Prathit Kurup, Victoria Figueroa

Company-to-company similarity network based on bill portfolio budget vectors.

Similarity metric -- cosine similarity of portfolio-share (frac) vectors:

  cos(i, j) = (u_i . u_j) / (||u_i|| * ||u_j||)

  where u_i[b] = firm i's spend on bill b / firm i's total lobbying budget.

Since all fracs are non-negative, cosine similarity is in [0, 1]:
  0 = no shared bills at all.
  1 = identical portfolio-share distribution (same bills, same proportions).

Relationship to BC similarity:
  BC similarity (bill_similarity_network) also uses frac vectors but computes
  breadth x depth (a non-linear composite). Cosine is a geometric angle measure
  that treats the full bill vocabulary as a single high-dimensional space.
  Both capture portfolio alignment; cosine is a linear inner product and does not
  separately reward breadth. The two networks are expected to be highly
  correlated but not identical, with cosine giving more weight to directional
  alignment across all shared bills rather than per-bill depth.

Two-stage filtering:
  Fracs are computed on ALL bills to preserve total-budget meaning.
  Mega-bills are excluded from the frac matrix before computing similarity,
  removing structurally uninformative shared dimensions.

Comparison note:
  All three similarity networks (bill_similarity_network, rbo_similarity_network,
  cosine_similarity_network) export CSV files to DATA_DIR with matching 'firm'
  column names so centrality tables can be joined and compared directly.

Usage:
  python cosine_similarity_network.py [options]

  --top-k K          Top-K subgraph for plot (default 20; 0 = skip plot)
  --centrality-k K   Firms printed per centrality metric (default 10; 0 = skip)
  --no-gml           Do not write a GML file
  --no-sweep         Skip the Leiden resolution sweep
  --resolution gamma Leiden resolution parameter (default 1.0)
  --min-weight FLOAT Minimum cosine similarity to include as edge (default 0.10)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import (aggregate_per_firm_bill, compute_zero_budget_fracs,
                               build_frac_matrix)
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import (compute_community_centralities,
                               print_community_centralities)
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "cosine_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "cosine_similarity_network.png")

LEIDEN_RESOLUTION  = 1.0
DEFAULT_MIN_WEIGHT = 0.10   # cosine is in [0,1]; 0.10 keeps meaningful edges


# -- Edge construction ---------------------------------------------------------

def company_cosine_edges(df, max_bill_df=MAX_BILL_DF,
                         min_weight=DEFAULT_MIN_WEIGHT):
    """
    Build cosine similarity edges between all firm pairs.

    Steps:
      1. Aggregate to one row per (firm, bill) by summing amounts.
      2. Compute fracs (portfolio shares) on the full bill universe.
      3. Apply prevalence filter to exclude mega-bills (same rationale as
         bill_similarity_network: they add a dominant shared dimension that
         reduces all pairwise cosines toward a common mean).
      4. Build firms x bills frac matrix (zero for unlobbied bills).
      5. Compute full pairwise cosine similarity via sklearn (vectorized BLAS).
      6. Extract upper triangle; keep edges >= min_weight.

    Returns DataFrame with columns [source, target, weight].
    """
    df_agg = aggregate_per_firm_bill(df)
    df_agg = compute_zero_budget_fracs(df_agg)  # adds frac, drops zero-budget firms

    # Two-stage filtering: fracs computed on ALL bills above. Now filter
    # mega-bills before building the similarity matrix.
    df_for_matrix = df_agg
    if max_bill_df is not None:
        df_for_matrix = filter_bills_by_prevalence(
            df_agg, max_bill_df, unit_col="bill_id"
        )

    pivot, firms, bills = build_frac_matrix(df_for_matrix)
    print(f"  Frac matrix: {len(firms):,} firms x {len(bills):,} bills  "
          f"(sparsity: {100*(pivot.values == 0).mean():.1f}% zeros)")

    mat    = pivot.values.astype(np.float64)    # shape (n_firms, n_bills)
    sim    = cosine_similarity(mat)             # shape (n_firms, n_firms)

    # Extract upper triangle (i < j) -- diagonal is 1.0 (self-similarity)
    n = len(firms)
    records = []
    rows, cols = np.triu_indices(n, k=1)
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


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Fortune 500 cosine similarity network (budget vectors)."
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
    p.add_argument("--min-weight",   type=float, default=DEFAULT_MIN_WEIGHT,
                   help=f"Minimum cosine similarity to include as edge "
                        f"(default {DEFAULT_MIN_WEIGHT}).")
    return p.parse_args()


# -- Main ----------------------------------------------------------------------

def main(args):
    df_raw = load_bills_data(DATA_DIR / "fortune500_lda_reports.csv")

    print(f"Bills: {df_raw['bill_id'].nunique():,}  |  "
          f"Companies: {df_raw['client_name'].nunique():,}")
    print(f"Cosine params: min_weight={args.min_weight}")

    edges = company_cosine_edges(df_raw, min_weight=args.min_weight)
    edge_weight_stats(edges, "cosine similarity (budget vectors)")
    edges.to_csv(DATA_DIR / "cosine_edges.csv", index=False)
    print(f"  Cosine edges -> {DATA_DIR / 'cosine_edges.csv'}")

    G = build_graph(edges)

    # -- Leiden community detection --------------------------------------------
    if not args.no_sweep:
        print("\n-- Resolution sweep (cosine similarity) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=args.resolution, seed=42
    )
    print(f"\n  Leiden (gamma={args.resolution})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="cosine similarity")

    comm_df = (
        pd.DataFrame([{"client_name": n, "community_cosine": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_cosine", "client_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_cosine.csv", index=False)
    print(f"\n  Community assignments -> {DATA_DIR / 'communities_cosine.csv'}")

    # -- Centrality (computed before GML so attrs can be embedded) -------------
    cent_df = None
    if not args.no_gml or args.centrality_k > 0:
        cent_df = compute_community_centralities(G, partition)
        if args.centrality_k > 0:
            print_community_centralities(cent_df, k=args.centrality_k)
            cent_df.to_csv(DATA_DIR / "centrality_cosine.csv", index=False)
            print(f"\n  Centrality -> {DATA_DIR / 'centrality_cosine.csv'}")

    # -- GML with enriched node attributes ------------------------------------
    if not args.no_gml:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
    # -------------------------------------------------------------------------

    if args.top_k > 0:
        H = top_k_subgraph(G, k=args.top_k)
        plot_circular(H,
                      title=f"Top {args.top_k} Fortune 500 Cosine Similarity Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main(parse_args())
