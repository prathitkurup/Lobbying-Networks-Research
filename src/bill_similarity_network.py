"""
Prathit Kurup, Victoria Figueroa

Company-to-company similarity network based on shared bill lobbying.

Similarity metric -- breadth x depth (Bray-Curtis):

  depth(i,j)   = mean BC similarity over shared bills
               = (1/|B_ij|) * sum_b [1 - |f_ib - f_jb| / (f_ib + f_jb)]

  breadth(i,j) = 1 - exp(-lambda * |B_ij|)

  weight(i,j)  = breadth(i,j) * depth(i,j)

where f_ib = firm i's total allocated spend on bill b / firm i's total budget.

Using each firm's own total budget as the denominator captures strategic
alignment: two firms score high when they allocate similar proportions of
their budgets to the same bills, regardless of absolute spending magnitude.

The breadth term rewards firms that coordinate across many bills (not just
one), using a saturating exponential so breadth approaches 1 asymptotically.
Lambda is set so breadth = 0.5 at the median shared-bill count; this is a
natural, data-adaptive calibration. See bc_diagnostics.robustness_check() to
verify results are insensitive to lambda choice.

Two-stage filtering:
  Fracs are computed on ALL bills (denominator = total budget unchanged).
  Mega-bills are excluded only from the pairing loop (MAX_BILL_DF from config).
  This removes spurious BC ~ 1.0 from tiny equal-frac omnibus bill allocations
  while preserving the economic meaning of frac as portfolio share.

Usage:
  python bill_similarity_network.py [options]

  --top-k K          Top-K subgraph for plot (default 20; 0 = skip plot)
  --centrality-k K   Firms printed per centrality metric (default 10; 0 = skip)
  --no-gml           Do not write a GML file
  --no-sweep         Skip the Leiden resolution sweep
  --resolution γ     Leiden resolution parameter (default 1.0)
  --lam FLOAT        Override lambda (default: auto log(2)/median_overlap)
  --diagnostics      Run full BC diagnostic suite after building edges
"""

import argparse
import sys
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import (compute_community_centralities,
                               print_community_centralities)
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "bill_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "bill_similarity_network.png")

LEIDEN_RESOLUTION = 1.0


# -- Edge construction --

def aggregate_per_firm_bill(df):
    """
    Collapse multiple rows per (client_name, bill_id) into one by summing
    amounts. extraction.py splits report-level spend equally across all bills
    in a report, so a firm that files multiple reports citing the same bill
    accumulates multiple rows — each carrying only a fraction of that report's
    spend. Summing gives the true total allocated spend per (firm, bill).
    """
    return df.groupby(["client_name", "bill_id"], as_index=False)["amount"].sum()


def company_bill_edges(df, lam=None, max_bill_df=MAX_BILL_DF):
    """
    Build breadth x depth similarity edges.

    Returns DataFrame with columns:
      [source, target, weight, shared_bills, depth, breadth]
    Only weight is passed to the graph builder; the rest are diagnostics.
    """
    df = aggregate_per_firm_bill(df)

    company_totals = df.groupby("client_name")["amount"].sum().rename("total_budget")
    df = df.merge(company_totals, on="client_name", how="left")

    # Exclude firms with zero total budget: they produce NaN fracs and silently
    # corrupt similarity scores. These are valid LDA filings with $0 spend.
    zero_budget = company_totals[company_totals == 0].index.tolist()
    if zero_budget:
        print(f"Warning: {len(zero_budget)} firm(s) excluded (zero budget): "
              f"{zero_budget}")
        df = df[df["total_budget"] > 0].copy()

    df["frac"] = df["amount"] / df["total_budget"]

    # Sanity check: fracs must sum to 1.0 per firm.
    frac_sums = df.groupby("client_name")["frac"].sum()
    bad = frac_sums[~frac_sums.between(0.999, 1.001)]
    if not bad.empty:
        raise ValueError(
            f"frac values do not sum to 1.0 for {len(bad)} firm(s): "
            f"{bad.to_dict()}\n"
            "Check extraction.py for changes that may have broken allocation logic."
        )

    # Two-stage filtering: fracs computed above on ALL bills (preserves denominator
    # meaning). Mega-bills excluded only from pairing loop below.
    df_for_pairing = df
    if max_bill_df is not None:
        df_for_pairing = filter_bills_by_prevalence(df, max_bill_df, unit_col="bill_id")

    bill_companies = (
        df_for_pairing.groupby("bill_id")
                      .apply(lambda x: list(zip(x["client_name"], x["frac"])),
                             include_groups=False)
    )

    records = []
    for bill_id, companies in bill_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                c1, f1 = companies[i]
                c2, f2 = companies[j]
                if c1 != c2 and (f1 + f2) > 0:
                    bc = 1 - abs(f1 - f2) / (f1 + f2)
                    src, tgt = (c1, c2) if c1 < c2 else (c2, c1)
                    records.append({"source": src, "target": tgt, "bc": bc})

    if not records:
        return pd.DataFrame(columns=["source", "target", "weight",
                                       "shared_bills", "depth", "breadth"])
    raw = pd.DataFrame(records).groupby(["source", "target"])
    agg = pd.concat([
        raw["bc"].mean().rename("depth"),
        raw["bc"].count().rename("shared_bills"),
    ], axis=1).reset_index()

    if lam is None:
        median_overlap = agg["shared_bills"].median()
        lam = np.log(2) / median_overlap
        print(f"\n  lambda auto-set to {lam:.5f}  "
              f"(breadth = 0.5 at median overlap = {median_overlap:.1f} bills)")

    agg["breadth"] = 1 - np.exp(-lam * agg["shared_bills"])
    agg["weight"]  = agg["breadth"] * agg["depth"]
    agg["weight"]  = pd.to_numeric(agg["weight"], errors="coerce")

    return agg[agg["weight"] > 0][
        ["source", "target", "weight", "shared_bills", "depth", "breadth"]
    ]


# -- CLI --

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Fortune 500 BC similarity network (breadth x depth)."
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
    p.add_argument("--lam",          type=float, default=None,
                   help="Override lambda (default: auto log(2)/median_overlap).")
    p.add_argument("--diagnostics",  action="store_true",
                   help="Run duplicate check, diagnostic summary, top-edges, and "
                        "lambda robustness check after building edges.")
    return p.parse_args()


# -- Main --

def main(args):
    from utils.bc_diagnostics import (duplicate_check, diagnostic_summary,
                                      top_edges_inspection, robustness_check)

    df_raw = load_bills_data(DATA_DIR / "fortune500_lda_reports.csv")

    if args.diagnostics:
        duplicate_check(df_raw)

    edges = company_bill_edges(df_raw, lam=args.lam)

    print(f"\nBills: {df_raw['bill_id'].nunique():,}  |  "
          f"Companies: {df_raw['client_name'].nunique():,}")
    edge_weight_stats(edges, "Breadth x Depth (Bray-Curtis)")

    if args.diagnostics:
        diagnostic_summary(edges)
        top_edges_inspection(edges, k=20)
        median_overlap = edges["shared_bills"].median()
        base_lam = np.log(2) / median_overlap
        robustness_check(company_bill_edges, df_raw,
                         lambdas=[base_lam * 0.5, base_lam * 2.0, base_lam * 4.0])

    edges[["source", "target", "weight"]].to_csv(
        DATA_DIR / "bc_edges.csv", index=False
    )
    print(f"  BC edges -> {DATA_DIR / 'bc_edges.csv'}")

    G = build_graph(edges[["source", "target", "weight"]])

    # -- Leiden community detection --
    if not args.no_sweep:
        print(f"\n-- Resolution sweep (BC similarity) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=args.resolution, seed=42
    )
    print(f"\n  Leiden (gamma={args.resolution})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="BC similarity")

    comm_df = (
        pd.DataFrame([{"client_name": n, "community_bc": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_bc", "client_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_bc.csv", index=False)
    print(f"\nCommunity assignments -> {DATA_DIR / 'communities_bc.csv'}")

    # -- Centrality (computed before GML so attrs can be embedded) --
    cent_df = None
    if not args.no_gml or args.centrality_k > 0:
        cent_df = compute_community_centralities(G, partition)
        if args.centrality_k > 0:
            print_community_centralities(cent_df, k=args.centrality_k)
            cent_df.to_csv(DATA_DIR / "centrality_bc.csv", index=False)
            print(f"\nCentrality -> {DATA_DIR / 'centrality_bc.csv'}")

    # -- GML with enriched node attributes --
    if not args.no_gml:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
    # --

    if args.top_k > 0:
        H = top_k_subgraph(G, k=args.top_k)
        plot_circular(H,
                      title=f"Top {args.top_k} Fortune 500 BC Similarity Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main(parse_args())
