"""
Company-to-company similarity network based on bill lobbying priority rankings.

Outputs: rbo_edges.csv, communities_rbo.csv,
centrality_rbo.csv, and GML visualization.
"""

import sys
import itertools
import pandas as pd
import networkx as nx

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import aggregate_per_firm_bill, compute_zero_budget_fracs, build_ranked_lists, rbo_score
from utils.network_building import build_graph, write_gml_with_communities, _cent_df_to_attrs, top_k_subgraph, edge_weight_stats
from utils.visualization import plot_circular
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.community import detect_communities, print_community_summary, sweep_resolution

GML_PATH = str(ROOT / "visualizations" / "gml" / "rbo_similarity_network.gml")
PNG_PATH = str(ROOT / "visualizations" / "png" / "rbo_similarity_network.png")

TOP_K           = 20
CENTRALITY_K    = 10
WRITE_GML       = True
RUN_SWEEP       = False
RBO_P           = 0.85
TOP_BILLS       = 30
MIN_WEIGHT      = 0.0
LEIDEN_RESOLUTION = 0.75


def company_rbo_edges(df, p=RBO_P, top_bills=TOP_BILLS,
                      max_bill_df=MAX_BILL_DF, min_weight=MIN_WEIGHT):
    """Build RBO similarity edges between all firm pairs."""
    df_agg = aggregate_per_firm_bill(df)
    df_agg = compute_zero_budget_fracs(df_agg)  # adds frac, drops zero-budget firms

    # Apply prevalence filter ONLY for building ranked lists (two-stage filter:
        # fracs computed on all bills above, mega-bills excluded here only)
    df_for_ranking = df_agg
    if max_bill_df is not None:
        df_for_ranking = filter_bills_by_prevalence(
            df_agg, max_bill_df, unit_col="bill_number"
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


def main():
    df_raw = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")

    print(f"Bills: {df_raw['bill_number'].nunique():,}  |  "
          f"Companies: {df_raw['fortune_name'].nunique():,}")
    print(f"RBO params: p={RBO_P}  top_bills={TOP_BILLS}  "
          f"min_weight={MIN_WEIGHT}")

    edges = company_rbo_edges(
        df_raw,
        p=RBO_P,
        top_bills=TOP_BILLS,
        min_weight=MIN_WEIGHT,
    )
    edge_weight_stats(edges, f"RBO similarity (p={RBO_P})")
    edges.to_csv(DATA_DIR / "rbo_edges.csv", index=False)
    print(f"\nRBO edges -> {DATA_DIR / 'rbo_edges.csv'}")

    G = build_graph(edges)

    if RUN_SWEEP:
        print("\n-- Resolution sweep (RBO similarity) --")
        sweep_resolution(G, resolutions=[0.7, 0.75, 0.8, 0.85, 0.9, 1.2], seed=42)

    partition, Q, comm_summary = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=42)
    print(f"\nLeiden (gamma={LEIDEN_RESOLUTION})  Q={Q:.4f}")
    print_community_summary(comm_summary, partition, G, label="RBO similarity")

    comm_df = (
        pd.DataFrame([{"fortune_name": n, "community_rbo": cid} for n, cid in partition.items()])
        .sort_values(["community_rbo", "fortune_name"])
        .reset_index(drop=True)
    )
    comm_df.to_csv(DATA_DIR / "communities_rbo.csv", index=False)
    print(f"\nCommunity assignments -> {DATA_DIR / 'communities_rbo.csv'}")

    cent_df = None
    if WRITE_GML or CENTRALITY_K > 0:
        cent_df = compute_community_centralities(G, partition)
        if CENTRALITY_K > 0:
            print_community_centralities(cent_df, k=CENTRALITY_K)
            cent_df.to_csv(DATA_DIR / "centrality_rbo.csv", index=False)
            print(f"\nCentrality -> {DATA_DIR / 'centrality_rbo.csv'}")

    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        write_gml_with_communities(G, partition, GML_PATH, node_attrs)

    if TOP_K > 0:
        H = top_k_subgraph(G, k=TOP_K)
        plot_circular(H,
                      title=f"Top {TOP_K} Fortune 500 RBO Similarity Network",
                      path=PNG_PATH)


if __name__ == "__main__":
    main()
