"""
Validation: centrality measures vs. agenda-setter rankings (116th Congress).

Compares four centrality measures from the bill affiliation network —
  BCZ intercentrality, global PageRank, within-community eigenvector centrality,
  and within-community PageRank —
against three directed-influence measures from the RBO directed network —
  net_influence, net_strength, and within-community net_influence/net_strength.

All within-community measures use stored Leiden community labels
(communities_affiliation.csv). BCZ intercentrality is computed fresh on the
affiliation graph following Ballester, Calvó-Armengol & Zenou (2006):
  bcz_i = sum(b(lambda, A)) - sum(b(lambda, A[-i]))
where b(lambda, A) is the Katz centrality vector, alpha = 0.85 / spectral_radius.

Comparisons use Spearman rank correlation on the full firm set and on the
top-N (N=30) by each measure. Results are written to a CSV and a summary txt.

Run from src/ directory:
  python validations/13_centrality_vs_agenda_setter.py
"""

import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT
from utils.centrality import compute_katz_centrality

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUT_DIR  = ROOT / "outputs" / "validation"
CSV_PATH = OUT_DIR / "13_centrality_vs_agenda_setter.csv"
TXT_PATH = OUT_DIR / "13_centrality_vs_agenda_setter.txt"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

TOP_N      = 30     # top firms per measure for overlap/rank comparison
WEIGHT_COL = "weight"

# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    """Write to stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_affiliation_graph():
    """Build undirected affiliation graph from stored edge file."""
    edges = pd.read_csv(DATA_DIR / "archive" / "network_edges" / "affiliation_edges.csv")
    G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr=WEIGHT_COL)
    return G


def load_community_partition():
    """Return {firm: community_id} from communities_affiliation.csv."""
    df = pd.read_csv(DATA_DIR / "archive" / "communities" / "communities_affiliation.csv")
    return dict(zip(df["fortune_name"], df["community_aff"]))


def load_centrality_affiliation():
    """Load precomputed affiliation centrality table."""
    return pd.read_csv(DATA_DIR / "archive" / "centralities" / "centrality_affiliation.csv")


def load_directed_influence():
    """Load 116th congress node attributes (net_influence, net_strength)."""
    return pd.read_csv(DATA_DIR / "congress" / "116" / "node_attributes.csv")


# ---------------------------------------------------------------------------
# BCZ intercentrality
# ---------------------------------------------------------------------------

def _katz_sum(G, alpha):
    """Total Katz centrality (sum over all nodes) for graph G at given alpha."""
    try:
        katz = nx.katz_centrality(G, alpha=alpha, beta=1.0, weight=WEIGHT_COL,
                                  max_iter=5000, normalized=False)
    except nx.PowerIterationFailedConvergence:
        # Fallback: weighted degree as proxy
        deg = dict(G.degree(weight=WEIGHT_COL))
        katz = deg
    return katz


def compute_bcz_intercentrality(G):
    """
    BCZ intercentrality for each node.
    bcz_i = sum(b(A)) - sum(b(A[-i])) where b is unnormalized Katz vector.
    Alpha set to 0.85 / spectral_radius on the full graph and reused for
    subgraphs (per BCZ: same decay parameter throughout).
    """
    A = nx.to_numpy_array(G, weight=WEIGHT_COL)
    try:
        eigs = np.linalg.eigvals(A)
        rho  = float(np.max(np.abs(eigs)))
    except np.linalg.LinAlgError:
        rho = 1.0
    alpha = 0.85 / rho if rho > 0 else 0.01

    # Full-graph Katz sum
    full_katz = _katz_sum(G, alpha)
    full_sum  = sum(full_katz.values())

    nodes = list(G.nodes())
    bcz   = {}
    for node in nodes:
        G_minus = G.copy()
        G_minus.remove_node(node)
        if G_minus.number_of_nodes() == 0:
            bcz[node] = full_sum
            continue
        reduced_katz = _katz_sum(G_minus, alpha)
        bcz[node] = full_sum - sum(reduced_katz.values())
    return bcz


# ---------------------------------------------------------------------------
# Within-community PageRank
# ---------------------------------------------------------------------------

def compute_within_community_pagerank(G, partition):
    """PageRank computed on each community subgraph independently."""
    comm_map = {}
    for node, cid in partition.items():
        comm_map.setdefault(cid, []).append(node)

    pr = {}
    for cid, members in comm_map.items():
        sub = G.subgraph(members)
        if sub.number_of_nodes() < 2:
            pr.update({n: 0.0 for n in members})
        else:
            pr.update(nx.pagerank(sub, weight=WEIGHT_COL))
    return pr


# ---------------------------------------------------------------------------
# Within-community directed measures
# ---------------------------------------------------------------------------

def compute_within_community_directed(nodes_df, partition):
    """
    Compute within-community net_influence and net_strength from directed
    edge file (116th congress). Uses out-edges only (both directions present);
    wc_net_strength = Σ_j∈same_community [RBO(i,j) × net_temporal(i,j)].
    """
    all_edges = pd.read_csv(DATA_DIR / "congress" / "116" / "rbo_directed_influence.csv")

    # Map each firm to community
    def comm(f):
        return partition.get(f)

    results = {}
    firms_in_nodes = set(nodes_df["firm"].tolist())

    for firm in firms_in_nodes:
        my_comm = comm(firm)
        if my_comm is None:
            results[firm] = {"wc_net_influence": np.nan, "wc_net_strength": np.nan}
            continue

        # Out-edges where firm is source and target is in same community
        # (both directions exist in CSV; out-edges cover all pairs without double-counting)
        same_comm_out = all_edges[
            (all_edges["source"] == firm) &
            (all_edges["target"].map(comm) == my_comm)
        ]

        wc_ni = int(
            same_comm_out["source_firsts"].sum() - same_comm_out["target_firsts"].sum()
        )
        wc_ns = float((same_comm_out["rbo"] * same_comm_out["net_temporal"]).sum())
        results[firm] = {"wc_net_influence": wc_ni, "wc_net_strength": wc_ns}

    return pd.DataFrame(results).T.rename_axis("firm").reset_index()


# ---------------------------------------------------------------------------
# Rank correlation helpers
# ---------------------------------------------------------------------------

def spearman(s1, s2, label_a, label_b, n=None):
    """
    Spearman rho and p-value between two Series aligned on index.
    If n is given, restrict to top-n by s1 before correlating.
    Returns (rho, pval, n_firms).
    """
    merged = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    if n is not None:
        merged = merged.nlargest(n, "a")
    if len(merged) < 5:
        return np.nan, np.nan, len(merged)
    rho, pval = spearmanr(merged["a"], merged["b"])
    return round(rho, 4), round(pval, 5), len(merged)


def top_n_overlap(s1, s2, n=TOP_N):
    """Fraction of top-n firms (by s1) that also appear in top-n by s2."""
    t1 = set(s1.nlargest(n).index)
    t2 = set(s2.nlargest(n).index)
    return round(len(t1 & t2) / n, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 70)
    print("VALIDATION 13: CENTRALITY VS. AGENDA-SETTER (116th CONGRESS)")
    print("=" * 70)
    print(f"\nAffiliation network → BCZ intercentrality, global PageRank,")
    print(f"within-community eigenvector, within-community PageRank.")
    print(f"Directed influence → net_influence, net_strength,")
    print(f"within-community net_influence, within-community net_strength.")
    print(f"\nTop-N window for overlap and restricted Spearman: N = {TOP_N}")

    # -- Load data -----------------------------------------------------------

    print("\n[1/6] Loading affiliation graph ...")
    G = load_affiliation_graph()
    print(f"      Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    partition = load_community_partition()
    print(f"      Communities: {len(set(partition.values()))}")

    cent_aff  = load_centrality_affiliation()
    nodes_116 = load_directed_influence()

    print(f"      Centrality table firms: {len(cent_aff)}")
    print(f"      116th node attrs firms: {len(nodes_116)}")

    # -- BCZ intercentrality -------------------------------------------------

    print("\n[2/6] Computing BCZ intercentrality (n-1 Katz subgraphs) ...")
    print(f"      This iterates over {G.number_of_nodes()} node removals — may take ~1 min.")
    bcz = compute_bcz_intercentrality(G)
    bcz_s = pd.Series(bcz, name="bcz_intercentrality")
    print(f"      Done. BCZ range: [{bcz_s.min():.4f}, {bcz_s.max():.4f}]")

    # -- Within-community PageRank -------------------------------------------

    print("\n[3/6] Computing within-community PageRank ...")
    wc_pr = compute_within_community_pagerank(G, partition)
    wc_pr_s = pd.Series(wc_pr, name="wc_pagerank")

    # -- Build master firm table ---------------------------------------------

    print("\n[4/6] Building master firm table ...")

    # Centrality measures (affiliation network)
    cent_idx = cent_aff.set_index("firm")
    global_pr_s    = cent_idx["global_pagerank"].rename("global_pagerank")
    wc_eigen_s     = cent_idx["within_comm_eigenvector"].rename("wc_eigenvector")
    katz_stored_s  = cent_idx["katz_centrality"].rename("katz_stored")

    # Directed measures (116th)
    nodes_idx = nodes_116.set_index("firm")
    net_inf_s  = nodes_idx["net_influence"].rename("net_influence")
    net_str_s  = nodes_idx["net_strength"].rename("net_strength")

    # Within-community directed measures
    print("      Computing within-community directed measures ...")
    wc_directed = compute_within_community_directed(nodes_116, partition)
    wc_directed = wc_directed.set_index("firm")
    wc_ni_s = wc_directed["wc_net_influence"].rename("wc_net_influence")
    wc_ns_s = wc_directed["wc_net_strength"].rename("wc_net_strength")

    # Merge all
    master = pd.concat([
        bcz_s, global_pr_s, wc_eigen_s, wc_pr_s,
        net_inf_s, net_str_s, wc_ni_s, wc_ns_s,
    ], axis=1)
    master.index.name = "firm"
    master = master.reset_index()

    print(f"      Master table: {len(master)} firms, "
          f"{master.dropna().shape[0]} with all measures complete.")

    # -- Spearman correlations -----------------------------------------------

    print("\n[5/6] Computing Spearman correlations ...")

    # Centrality measures
    centrality_measures = {
        "bcz_intercentrality":  bcz_s,
        "global_pagerank":      global_pr_s,
        "wc_eigenvector":       wc_eigen_s,
        "wc_pagerank":          wc_pr_s,
    }
    # Agenda-setter measures (global vs. within-community)
    agenda_measures_global = {
        "net_influence": net_inf_s,
        "net_strength":  net_str_s,
    }
    agenda_measures_wc = {
        "wc_net_influence": wc_ni_s,
        "wc_net_strength":  wc_ns_s,
    }

    records = []

    def _pair_row(cent_name, ag_name, cent_ser, ag_ser):
        """Build one row of results for a centrality–agenda pair."""
        rho_full, p_full, n_full = spearman(cent_ser, ag_ser, cent_name, ag_name)
        rho_top,  p_top,  n_top  = spearman(cent_ser, ag_ser, cent_name, ag_name, n=TOP_N)
        ov = top_n_overlap(cent_ser, ag_ser, n=TOP_N)
        return {
            "centrality_measure": cent_name,
            "agenda_measure":     ag_name,
            "spearman_rho_full":  rho_full,
            "pval_full":          p_full,
            "n_firms_full":       n_full,
            "spearman_rho_top30": rho_top,
            "pval_top30":         p_top,
            "n_firms_top30":      n_top,
            f"top{TOP_N}_overlap_fraction": ov,
        }

    # Global centrality vs. global agenda-setters
    for cn, cs in centrality_measures.items():
        for an, ag in agenda_measures_global.items():
            records.append(_pair_row(cn, an, cs, ag))

    # Within-community centrality vs. within-community agenda-setters
    for cn, cs in [("wc_eigenvector", wc_eigen_s), ("wc_pagerank", wc_pr_s)]:
        for an, ag in agenda_measures_wc.items():
            records.append(_pair_row(cn, an, cs, ag))

    # BCZ vs. within-community agenda-setters (cross check)
    for an, ag in agenda_measures_wc.items():
        records.append(_pair_row("bcz_intercentrality", an, bcz_s, ag))

    corr_df = pd.DataFrame(records)

    # -- Print results table -------------------------------------------------

    print("\n" + "-" * 70)
    print(f"{'Centrality':<25} {'Agenda measure':<22} "
          f"{'ρ_full':>7} {'p_full':>8} "
          f"{'ρ_top30':>8} {'p_top30':>8} "
          f"{'Overlap':>8}")
    print("-" * 70)
    for _, row in corr_df.iterrows():
        print(f"{row['centrality_measure']:<25} {row['agenda_measure']:<22} "
              f"{str(row['spearman_rho_full']):>7} {str(row['pval_full']):>8} "
              f"{str(row['spearman_rho_top30']):>8} {str(row['pval_top30']):>8} "
              f"{str(row[f'top{TOP_N}_overlap_fraction']):>8}")
    print("-" * 70)

    # -- Top-N ranked lists --------------------------------------------------

    print(f"\n[6/6] Top-{TOP_N} firm lists by each measure:")

    def _top_list(ser, label):
        top = ser.dropna().nlargest(TOP_N)
        print(f"\n  --- Top {TOP_N} by {label} ---")
        for rank, (firm, val) in enumerate(top.items(), 1):
            print(f"  {rank:>3}. {firm:<42} {val:>10.4f}")

    _top_list(bcz_s,        "BCZ Intercentrality")
    _top_list(global_pr_s,  "Global PageRank")
    _top_list(wc_eigen_s,   "Within-Community Eigenvector")
    _top_list(wc_pr_s,      "Within-Community PageRank")
    _top_list(net_str_s,    "Net Strength (directed, primary)")
    _top_list(wc_ns_s,      "Within-Community Net Strength")
    _top_list(net_inf_s,    "Net Influence (directed, reference)")
    _top_list(wc_ni_s,      "Within-Community Net Influence")

    # -- Interpretation flags ------------------------------------------------

    print("\n  --- Interpretation summary ---")
    # Primary: BCZ ↔ net_strength (RBO-weighted temporal dominance — BCZ theoretical bridge)
    bcz_ns = corr_df.loc[
        (corr_df["centrality_measure"] == "bcz_intercentrality") &
        (corr_df["agenda_measure"] == "net_strength"), "spearman_rho_full"
    ].values[0]
    bcz_ni = corr_df.loc[
        (corr_df["centrality_measure"] == "bcz_intercentrality") &
        (corr_df["agenda_measure"] == "net_influence"), "spearman_rho_full"
    ].values[0]
    print(f"\n  BCZ intercentrality ↔ net_strength [primary] (Spearman ρ = {bcz_ns}):")
    if abs(bcz_ns) >= 0.5:
        print("  STRONG alignment: structural key-players (BCZ) ≈ empirical "
              "agenda-setters by RBO-weighted temporal dominance. Supports BCZ bridge.")
    elif abs(bcz_ns) >= 0.3:
        print("  MODERATE alignment: partial overlap between BCZ key-players "
              "and net_strength agenda-setters.")
    else:
        print("  WEAK alignment: BCZ undirected position does not predict "
              "RBO-weighted temporal leadership (net_strength).")
    print(f"\n  BCZ intercentrality ↔ net_influence [reference] (Spearman ρ = {bcz_ni}):")

    # -- Save outputs --------------------------------------------------------

    # Master firm table
    master.to_csv(CSV_PATH, index=False)
    print(f"\n  Results CSV -> {CSV_PATH}")

    # Correlation table as separate CSV
    corr_csv = OUT_DIR / "13_centrality_vs_agenda_setter_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"  Correlation table -> {corr_csv}")

    print("\n  Validation complete.")
    print("=" * 70)

    log_f.close()
    sys.stdout = sys.__stdout__
    print(f"\nLog written to {TXT_PATH}")


if __name__ == "__main__":
    main()
