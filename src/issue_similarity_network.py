"""
Prathit Kurup, Victoria Figueroa

Company-to-company similarity network based on shared issue-code lobbying.

Bray-Curtis per issue:  BC(i,j) = 1 - |f_i - f_j| / (f_i + f_j)
  where f_i = firm i's spend on this issue / firm i's total lobbying budget.

Aggregated:  weight(i,j) = sum_b BC(i,j) / sqrt(shared_issue_count)  [NORMALIZE=True]
             weight(i,j) = sum_b BC(i,j)                             [NORMALIZE=False]

Issue codes are broader than bills (75 codes vs. 2300+ bills), so this network
captures strategic alignment at the policy-area level rather than bill-by-bill.

Prevalence filtering (MAX_ISSUE_DF):
  Disabled by default (None) since issue codes are broad by design and most
  firms lobby multiple areas. Set a threshold in config.py if modularity
  collapses — analogous to the bill-level mega-bill problem.

Usage:
  python issue_similarity_network.py [options]

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
from config import DATA_DIR, ROOT, MAX_ISSUE_DF
from utils.data_loading import load_issues_data
from utils.filtering import filter_bills_by_prevalence
from utils.network_building import (build_graph, write_gml_with_communities,
                                    _cent_df_to_attrs, top_k_subgraph,
                                    edge_weight_stats)
from utils.visualization import plot_circular
from utils.centrality import (compute_community_centralities,
                               print_community_centralities)
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "issue_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "issue_similarity_network.png")

NORMALIZE         = True   # divide raw BC sum by sqrt(shared issue count)
LEIDEN_RESOLUTION = 1.0


# -- Edge construction --

def company_issue_edges(df, normalize=NORMALIZE, max_issue_df=MAX_ISSUE_DF):
    """
    Build BC similarity edges where each firm's fractional spend on an issue
    is relative to that firm's total lobbying budget (not total issue spend).

    Data note:
      fortune500_lda_issues.csv has one row per (report, issue_code), so a firm
      filing multiple reports on the same issue has multiple rows. Without
      aggregation the clique loop generates a cartesian product of row counts
      per (firm, issue), producing identical inflation to the bill-level bug.
      Sum amounts to one row per (client_name, general_issue_code) first.

    Two-stage filtering (if max_issue_df set):
      Fracs are computed on ALL issue codes (preserves total-budget meaning).
      Filtering is applied only to the pairing loop.
    """
    df = df.groupby(["client_name", "general_issue_code"],
                    as_index=False)["amount"].sum()

    company_totals = df.groupby("client_name")["amount"].sum().rename("total_budget")
    df = df.merge(company_totals, on="client_name", how="left")

    zero_budget = company_totals[company_totals == 0].index.tolist()
    if zero_budget:
        print(f"  Warning: {len(zero_budget)} firm(s) excluded (zero budget): "
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

    df_for_pairing = df
    if max_issue_df is not None:
        df_for_pairing = filter_bills_by_prevalence(df, max_issue_df,
                                                    unit_col="general_issue_code")

    issue_companies = (
        df_for_pairing.groupby("general_issue_code")
                      .apply(lambda x: list(zip(x["client_name"], x["frac"])),
                             include_groups=False)
    )

    records = []
    for issue_code, companies in issue_companies.items():
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                c1, f1 = companies[i]
                c2, f2 = companies[j]
                if c1 != c2 and (f1 + f2) > 0:
                    bc = 1 - abs(f1 - f2) / (f1 + f2)
                    src, tgt = (c1, c2) if c1 < c2 else (c2, c1)
                    records.append({"source": src, "target": tgt, "weight": bc})

    if not records:
        return pd.DataFrame(columns=["source", "target", "weight"])
    edges = pd.DataFrame(records).groupby(["source", "target"])

    if normalize:
        weight_sum   = edges["weight"].sum()
        shared_count = edges["weight"].count().rename("shared_issues")
        result = pd.concat([weight_sum, shared_count], axis=1).reset_index()
        result["weight"] = result["weight"] / result["shared_issues"].pow(0.5)
        result = result[["source", "target", "weight"]]
    else:
        result = edges["weight"].sum().reset_index()

    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    return result[result["weight"] > 0]


# -- CLI --

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Fortune 500 issue-code BC similarity network."
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
    df    = load_issues_data(DATA_DIR / "fortune500_lda_issues.csv")
    edges = company_issue_edges(df)

    print(f"Issues: {df['general_issue_code'].nunique()}  |  "
          f"Companies: {df['client_name'].nunique():,}")
    edge_weight_stats(edges, f"BC issue similarity (normalize={NORMALIZE})")

    G = build_graph(edges)

    # -- Leiden community detection --
    if not args.no_sweep:
        print(f"\n-- Resolution sweep (issue similarity) --")
        sweep_resolution(G, resolutions=[0.5, 0.75, 1.0, 1.15, 1.25], seed=42)

    partition, Q, comm_summary = detect_communities(
        G, resolution=args.resolution, seed=42
    )
    print(f"\n  Leiden (gamma={args.resolution})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="issue similarity")

    comm_df = (
        pd.DataFrame([{"client_name": n, "community_issue": cid}
                      for n, cid in partition.items()])
        .sort_values(["community_issue", "client_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_issue.csv", index=False)
    print(f"  Community assignments -> {DATA_DIR / 'communities_issue.csv'}")

    # -- Centrality (computed before GML so attrs can be embedded) --
    cent_df = None
    if not args.no_gml or args.centrality_k > 0:
        cent_df = compute_community_centralities(G, partition)
        if args.centrality_k > 0:
            print_community_centralities(cent_df, k=args.centrality_k)
            cent_df.to_csv(DATA_DIR / "centrality_issue.csv", index=False)
            print(f"  Centrality -> {DATA_DIR / 'centrality_issue.csv'}")

    # -- GML with enriched node attributes --
    if not args.no_gml:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)
    # --

    if args.top_k > 0:
        H = top_k_subgraph(G, k=args.top_k)
        plot_circular(H,
                      title=f"Top {args.top_k} Fortune 500 Issue Similarity Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main(parse_args())
