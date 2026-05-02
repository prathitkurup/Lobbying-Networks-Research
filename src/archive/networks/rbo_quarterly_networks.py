"""
Quarter-by-quarter RBO similarity networks for the 116th Congress.

Builds 8 independent RBO networks (Q1 2019 → Q4 2020) and for each quarter:
  - Saves per-quarter edge CSV, community CSV, centrality CSV
  - Writes a Gephi-compatible GML with full node attributes
  - Saves a top-K circular PNG visualization
  - Prints network stats

Then runs three temporal evolution analyses:

  A) Community stability  — NMI + ARI between consecutive Leiden partitions.
     Tells you how much lobbying coalitions restructure each quarter.

  B) Network metric trajectories — density, mean RBO weight, weighted clustering
     coefficient, modularity Q, community count across all 8 quarters.
     Printed as a table; also saved to rbo_quarterly_stats.csv.

  C) Centrality rank stability — Spearman ρ of PageRank between consecutive
     quarters over the intersection of firms present in both. Tells you whether
     the same firms persistently dominate the influence hierarchy.

Output files (see DATA_DIR and VIZ_DIR below):
  rbo_edges_q{1..8}.csv                per-quarter edge lists
  communities_rbo_q{1..8}.csv          per-quarter Leiden community assignments
  centrality_rbo_q{1..8}.csv           per-quarter full centrality table
  rbo_quarterly_stats.csv              8-row network metric table
  rbo_quarterly_nmi_ari.csv            7-row temporal analysis A results
  rbo_quarterly_spearman.csv           7-row temporal analysis C results
  visualizations/gml/rbo_q{1..8}.gml   Gephi GML with full node attributes
  visualizations/png/rbo_q{1..8}.png   Top-K circular layout plots

Design notes (see design_decisions.md §19):
  - Each quarter is an independent window (no rolling/cumulative aggregation).
  - RBO params (p, top_bills) and Leiden resolution are identical to the
    full-congress rbo_similarity_network.py for comparability.
  - MAX_BILL_DF (=50) is applied per quarter; per-quarter prevalence will be
    lower, so fewer bills are filtered — this is correct behavior.
  - For temporal analyses, only firms present in both compared quarters are used.
"""

import sys
import pandas as pd
import networkx as nx
from scipy import stats as scipy_stats
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.network_building import (
    build_graph, write_gml_with_communities,
    _cent_df_to_attrs, top_k_subgraph,
)
from utils.community import detect_communities
from utils.centrality import compute_community_centralities, print_community_centralities
from utils.visualization import plot_circular
from rbo_similarity_network import company_rbo_edges

# -- Output directories --------------------------------------------------------
VIZ_GML_DIR = ROOT / "visualizations" / "gml"
VIZ_PNG_DIR = ROOT / "visualizations" / "png"

# -- Parameters (match full-congress script for comparability) -----------------
RBO_P             = 0.85
TOP_BILLS         = 30
MIN_WEIGHT        = 0.0
LEIDEN_RESOLUTION = 0.75
TOP_K             = 20      # firms in PNG circular plot (by weighted degree)
CENTRALITY_K      = 10      # top-k printed in centrality summary
WRITE_CSVS        = True
WRITE_GML         = True
WRITE_PNG         = True

QUARTER_LABELS = {
    1: "2019 Q1", 2: "2019 Q2", 3: "2019 Q3", 4: "2019 Q4",
    5: "2020 Q1", 6: "2020 Q2", 7: "2020 Q3", 8: "2020 Q4",
}


# -- Quarter assignment --------------------------------------------------------

def assign_quarters(df):
    """Add quarter column (1-8): base quarter from report_type prefix + year offset.

    report_type variants (q1/q1a/q1t/q1ta, etc.) are all mapped to the same
    base quarter number for that reporting period.
    """
    base_q   = df["report_type"].str[:2].str[1].astype(int)  # 1, 2, 3, 4
    year_off = df["year"].map({2019: 0, 2020: 4})
    df       = df.copy()
    df["quarter"] = base_q + year_off
    return df


# -- Per-quarter network build -------------------------------------------------

def build_quarter_network(df_all, quarter):
    """Build RBO network, centralities, GML, and PNG for a single quarter.

    Filters df_all to the given quarter, calls company_rbo_edges(), builds the
    NetworkX graph, runs Leiden, computes full centrality suite, writes GML and
    PNG outputs, and returns the key outputs for temporal analysis.

    Parameters
    ----------
    df_all  : full DataFrame (all quarters, already quarter-assigned)
    quarter : int 1-8

    Returns
    -------
    edges     : pd.DataFrame  edge list (source, target, weight)
    G         : nx.Graph
    partition : dict {firm: community_id}  (empty dict if no edges)
    s         : dict of network stats (empty dict if no edges)
    """
    label   = QUARTER_LABELS[quarter]
    df_q    = df_all[df_all["quarter"] == quarter].copy()
    n_firms = df_q["fortune_name"].nunique()
    n_bills = df_q["bill_number"].nunique()

    print(f"\n{'='*62}")
    print(f"  {label}  (quarter {quarter})")
    print(f"  Input rows: {len(df_q):,}  |  Firms: {n_firms}  |  Bills: {n_bills}")

    edges = company_rbo_edges(
        df_q,
        p           = RBO_P,
        top_bills   = TOP_BILLS,
        max_bill_df = MAX_BILL_DF,
        min_weight  = MIN_WEIGHT,
    )

    if edges.empty:
        print(f"  *** No edges produced for {label} ***")
        return edges, nx.Graph(), {}, {}

    G = build_graph(edges)

    n        = G.number_of_nodes()
    m        = G.number_of_edges()
    density  = nx.density(G)
    mean_w   = edges["weight"].mean()
    cluster  = nx.average_clustering(G, weight="weight")

    partition, Q_mod, comm_summary = detect_communities(
        G, resolution=LEIDEN_RESOLUTION, seed=42
    )
    n_comm = len(comm_summary)

    print(f"\n  Network stats:")
    print(f"    Nodes: {n}  |  Edges: {m:,}  |  Density: {density:.4f}")
    print(f"    Mean RBO weight: {mean_w:.4f}  |  Clustering: {cluster:.4f}")
    print(f"    Leiden communities: {n_comm}  |  Modularity Q: {Q_mod:.4f}")

    # -- Centralities --
    cent_df = compute_community_centralities(G, partition)
    if CENTRALITY_K > 0:
        print_community_centralities(cent_df, k=CENTRALITY_K)

    # -- Write per-quarter CSVs --
    if WRITE_CSVS:
        edges.to_csv(DATA_DIR / f"rbo_edges_q{quarter}.csv", index=False)

        comm_df = (
            pd.DataFrame([
                {"fortune_name": node, "community_rbo": cid, "quarter": quarter}
                for node, cid in partition.items()
            ])
            .sort_values(["community_rbo", "fortune_name"])
            .reset_index(drop=True)
        )
        comm_df.to_csv(DATA_DIR / f"communities_rbo_q{quarter}.csv", index=False)
        cent_df.to_csv(DATA_DIR / f"centrality_rbo_q{quarter}.csv", index=False)

    # -- Write GML --
    if WRITE_GML:
        node_attrs = _cent_df_to_attrs(cent_df) or {}
        node_attrs["kcore"] = nx.core_number(G)
        # Tag network_label so Gephi can identify the quarter in multi-file sessions
        node_attrs["network_label"] = {node: label for node in G.nodes()}
        gml_path = str(VIZ_GML_DIR / f"rbo_q{quarter}.gml")
        write_gml_with_communities(G, partition, gml_path, node_attrs)

    # -- Write PNG --
    if WRITE_PNG:
        H = top_k_subgraph(G, k=TOP_K)
        png_path = str(VIZ_PNG_DIR / f"rbo_q{quarter}.png")
        plot_circular(
            H,
            title=f"Top {TOP_K} Fortune 500 RBO Similarity — {label}",
            path=png_path,
        )

    s = {
        "quarter":       quarter,
        "label":         label,
        "nodes":         n,
        "edges":         m,
        "density":       density,
        "mean_weight":   mean_w,
        "clustering":    cluster,
        "modularity":    Q_mod,
        "n_communities": n_comm,
    }
    return edges, G, partition, s


# -- Temporal analysis A: community stability (NMI + ARI) ---------------------

def temporal_analysis_A(partitions):
    """NMI + ARI between Leiden partitions of consecutive quarters.

    Uses the intersection of firms present in both quarters.  NMI (arithmetic
    average method) and ARI are both label-permutation invariant — community
    IDs need not be aligned across runs.

    Interpretation guide:
      NMI > 0.70: high coalition stability
      NMI 0.40–0.70: moderate — some restructuring
      NMI < 0.40: low — coalitions substantially reorganize each quarter
    """
    print(f"\n{'='*62}")
    print("  TEMPORAL ANALYSIS A — Community Stability (NMI + ARI)")
    print("  NMI/ARI on shared firms between consecutive quarter partitions")
    print(f"{'='*62}")
    print(f"  {'Transition':<16}  {'NMI':>8}  {'ARI':>8}  {'Shared firms':>13}")
    print(f"  {'-'*50}")

    quarters = sorted(partitions.keys())
    results  = []

    for i in range(len(quarters) - 1):
        q1, q2 = quarters[i], quarters[i + 1]
        p1, p2 = partitions[q1], partitions[q2]
        shared = sorted(set(p1) & set(p2))

        if len(shared) < 2:
            print(f"  Q{q1}→Q{q2}  —  insufficient shared firms ({len(shared)})")
            continue

        labs1 = [p1[f] for f in shared]
        labs2 = [p2[f] for f in shared]
        nmi   = normalized_mutual_info_score(labs1, labs2, average_method="arithmetic")
        ari   = adjusted_rand_score(labs1, labs2)

        tag = f"Q{q1}→Q{q2}"
        print(f"  {tag:<16}  {nmi:>8.4f}  {ari:>8.4f}  {len(shared):>13,}")
        results.append({
            "transition": tag, "q_from": q1, "q_to": q2,
            "nmi": nmi, "ari": ari, "shared_firms": len(shared),
        })

    if results:
        avg_nmi = sum(r["nmi"] for r in results) / len(results)
        avg_ari = sum(r["ari"] for r in results) / len(results)
        print(f"\n  Mean NMI: {avg_nmi:.4f}  |  Mean ARI: {avg_ari:.4f}")
        if avg_nmi > 0.70:
            print("  → High coalition stability: lobbying coalitions are largely persistent.")
        elif avg_nmi > 0.40:
            print("  → Moderate stability: meaningful coalition restructuring each quarter.")
        else:
            print("  → Low stability: coalitions substantially reorganize quarter to quarter.")

    return results


# -- Temporal analysis B: network metric trajectories -------------------------

def temporal_analysis_B(stats_list):
    """Print the 8-quarter network metric trajectory table.

    Covers: nodes, edges, density, mean RBO weight, clustering coeff,
    modularity Q, community count.  Data are also in rbo_quarterly_stats.csv.
    """
    print(f"\n{'='*62}")
    print("  TEMPORAL ANALYSIS B — Network Metric Trajectories")
    print(f"{'='*62}")
    print(
        f"  {'Quarter':<10}  {'Nodes':>6}  {'Edges':>7}  {'Density':>8}  "
        f"{'MeanW':>7}  {'Cluster':>8}  {'ModQ':>6}  {'Comms':>6}"
    )
    print(f"  {'-'*72}")

    for s in stats_list:
        print(
            f"  {s['label']:<10}  {s['nodes']:>6}  {s['edges']:>7,}  "
            f"{s['density']:>8.4f}  {s['mean_weight']:>7.4f}  "
            f"{s['clustering']:>8.4f}  {s['modularity']:>6.4f}  "
            f"{s['n_communities']:>6}"
        )


# -- Temporal analysis C: centrality rank stability (Spearman ρ) --------------

def temporal_analysis_C(graphs):
    """Spearman ρ of PageRank between consecutive quarters.

    Computed over the intersection of firms in both quarters.  High ρ means
    the same firms are consistently the most influential across the two periods;
    low ρ signals churn in who holds structural power.

    Interpretation guide:
      ρ > 0.85: very stable influence hierarchy
      ρ 0.60–0.85: moderate — some rank reshuffling
      ρ < 0.60: high volatility — centrality rankings shift substantially
    """
    print(f"\n{'='*62}")
    print("  TEMPORAL ANALYSIS C — Centrality Rank Stability (Spearman ρ)")
    print("  PageRank on weighted graph; ρ computed over shared firms")
    print(f"{'='*62}")
    print(f"  {'Transition':<16}  {'Spearman ρ':>11}  {'p-value':>9}  {'Shared firms':>13}")
    print(f"  {'-'*54}")

    quarters = sorted(graphs.keys())
    results  = []

    for i in range(len(quarters) - 1):
        q1, q2 = quarters[i], quarters[i + 1]
        G1, G2 = graphs[q1], graphs[q2]

        if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
            continue

        pr1    = nx.pagerank(G1, weight="weight")
        pr2    = nx.pagerank(G2, weight="weight")
        shared = sorted(set(pr1) & set(pr2))

        if len(shared) < 3:
            print(f"  Q{q1}→Q{q2}  —  insufficient shared firms ({len(shared)})")
            continue

        v1        = [pr1[f] for f in shared]
        v2        = [pr2[f] for f in shared]
        rho, pval = scipy_stats.spearmanr(v1, v2)

        tag = f"Q{q1}→Q{q2}"
        print(f"  {tag:<16}  {rho:>11.4f}  {pval:>9.4f}  {len(shared):>13,}")
        results.append({
            "transition": tag, "q_from": q1, "q_to": q2,
            "spearman_rho": rho, "pvalue": pval, "shared_firms": len(shared),
        })

    if results:
        avg_rho = sum(r["spearman_rho"] for r in results) / len(results)
        print(f"\n  Mean Spearman ρ: {avg_rho:.4f}")
        if avg_rho > 0.85:
            print("  → Very stable influence hierarchy across quarters.")
        elif avg_rho > 0.60:
            print("  → Moderate influence stability: some churn in structural power.")
        else:
            print("  → High volatility: centrality rankings shift substantially each quarter.")

    return results


# -- Main ----------------------------------------------------------------------

def main():
    df_raw = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")

    # Assign quarters if the column is not already present
    if "quarter" not in df_raw.columns:
        df_raw = assign_quarters(df_raw)

    print(f"Loaded {len(df_raw):,} rows  |  "
          f"{df_raw['fortune_name'].nunique()} firms  |  "
          f"{df_raw['bill_number'].nunique()} bills")
    print(f"RBO params: p={RBO_P}  top_bills={TOP_BILLS}  "
          f"min_weight={MIN_WEIGHT}  leiden_resolution={LEIDEN_RESOLUTION}")

    stats_list = []
    partitions = {}   # {quarter: {firm: community_id}}
    graphs     = {}   # {quarter: nx.Graph}

    for q in range(1, 9):
        edges, G, partition, s = build_quarter_network(df_raw, q)

        if not edges.empty:
            stats_list.append(s)
            partitions[q] = partition
            graphs[q]     = G

    # -- Temporal analyses --
    a_results = temporal_analysis_A(partitions)
    temporal_analysis_B(stats_list)
    c_results = temporal_analysis_C(graphs)

    # -- Write summary CSVs --
    if WRITE_CSVS and stats_list:
        pd.DataFrame(stats_list).to_csv(
            DATA_DIR / "rbo_quarterly_stats.csv", index=False
        )
        print(f"\nStats       -> {DATA_DIR / 'rbo_quarterly_stats.csv'}")

    if WRITE_CSVS and a_results:
        pd.DataFrame(a_results).to_csv(
            DATA_DIR / "rbo_quarterly_nmi_ari.csv", index=False
        )
        print(f"NMI/ARI     -> {DATA_DIR / 'rbo_quarterly_nmi_ari.csv'}")

    if WRITE_CSVS and c_results:
        pd.DataFrame(c_results).to_csv(
            DATA_DIR / "rbo_quarterly_spearman.csv", index=False
        )
        print(f"Spearman ρ  -> {DATA_DIR / 'rbo_quarterly_spearman.csv'}")


if __name__ == "__main__":
    main()
