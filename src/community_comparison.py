"""
Prathit Kurup

Compare Leiden community partitions from the bill affiliation network and the
BC similarity network.

This script answers two questions:
  1. Agreement: how similar are the two partitions globally?
     → Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI)
  2. Firm-level stability: which firms are consistently placed in the same
     industry cluster regardless of which network you use, and which firms
     cross community boundaries — the potential cross-industry connectors?

Outputs
-------
  data/community_comparison.csv     — per-firm community assignments from both
                                      networks plus a stability flag
  data/community_confusion.csv      — community-level co-occurrence matrix
                                      (how affiliation communities map to BC
                                      communities)

Run: python community_comparison.py
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sys.path.insert(0, ".")
from config import DATA_DIR
from utils.data_loading import load_bills_data, load_issues_data
from utils.network_building import build_graph
from utils.community import (
    detect_communities, print_community_summary, sweep_resolution,
    community_sizes,
)

# -- Imports for network building --
from bill_affiliation_network import company_bill_edges
from bill_similarity_network import company_bill_edges as bc_bill_edges

# Leiden settings — must match the values in each network's main script so
# that comparisons reflect the canonical partitions, not arbitrary ones.
LEIDEN_RESOLUTION = 1.0
SEED              = 42

# Threshold for flagging a firm as a "cross-community mover" between the two
# network partitions (different community assignment in aff vs. BC network).
# All movers are flagged; this label is used in the output for easy filtering.
MOVER_LABEL  = "cross-community"
STABLE_LABEL = "stable"


# -- Build networks and detect communities --

def build_affiliation_graph():
    df    = load_bills_data(DATA_DIR / "fortune500_lda_reports.csv")
    edges = company_bill_edges(df)
    return build_graph(edges)


def build_bc_graph():
    df    = load_bills_data(DATA_DIR / "fortune500_lda_reports.csv")
    edges = bc_bill_edges(df, lam=None)
    return build_graph(edges[["source", "target", "weight"]])


# -- Comparison utilities --

def align_community_ids(part_a, part_b):
    """
    Re-label community IDs in part_b so that they are aligned with part_a
    using the Hungarian assignment (maximum overlap). Returns a new dict
    with the same keys as part_b but with relabeled community IDs.

    This makes the confusion matrix more readable: community 0 in the
    affiliation network will correspond to whichever BC community overlaps
    with it most.
    """
    from scipy.optimize import linear_sum_assignment

    ids_a = sorted(set(part_a.values()))
    ids_b = sorted(set(part_b.values()))

    # Build overlap matrix [i, j] = number of nodes in comm i (a) and comm j (b)
    overlap = np.zeros((len(ids_a), len(ids_b)), dtype=int)
    shared_nodes = set(part_a) & set(part_b)
    for node in shared_nodes:
        ia = ids_a.index(part_a[node])
        ib = ids_b.index(part_b[node])
        overlap[ia, ib] += 1

    # Hungarian algorithm on negative overlap (maximise)
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[ids_b[c]] = ids_a[r]
    # Any unmatched IDs in b get unique new IDs beyond the range of a
    next_id = max(ids_a) + 1
    for ib in ids_b:
        if ib not in mapping:
            mapping[ib] = next_id
            next_id += 1

    return {node: mapping[cid] for node, cid in part_b.items()}


def compute_global_agreement(part_a, part_b):
    """NMI and ARI between two partitions over their shared nodes."""
    shared = sorted(set(part_a) & set(part_b))
    labels_a = [part_a[n] for n in shared]
    labels_b = [part_b[n] for n in shared]
    nmi = normalized_mutual_info_score(labels_a, labels_b, average_method="arithmetic")
    ari = adjusted_rand_score(labels_a, labels_b)
    return nmi, ari, len(shared)


def confusion_matrix_df(part_a, part_b, label_a="affiliation", label_b="bc"):
    """
    Build a community co-occurrence table.
    Rows = affiliation communities, columns = BC communities.
    Cell (i, j) = number of firms assigned to community i in network A
    and community j in network B.
    """
    shared = set(part_a) & set(part_b)
    records = [(part_a[n], part_b[n]) for n in shared]
    df = pd.DataFrame(records, columns=[label_a, label_b])
    return df.pivot_table(index=label_a, columns=label_b,
                          aggfunc="size", fill_value=0)


def firm_stability_table(part_aff, part_bc, G_aff, G_bc):
    """
    Build a per-firm table with:
      client_name      : firm name
      community_aff    : Leiden community in bill affiliation network
      community_bc     : Leiden community in BC similarity network
      stability        : 'stable' if same community in both; 'cross-community' if not
      degree_aff       : weighted degree in affiliation network
      degree_bc        : weighted degree in BC network
      intra_edges_aff  : count of edges to own community (affiliation)
      intra_edges_bc   : count of edges to own community (BC)
      cross_edges_aff  : count of edges to other communities (affiliation)
      cross_edges_bc   : count of edges to other communities (BC)
      participation_aff: P = 1 - Σ_c (k_ic/k_i)^2 (affiliation)
      participation_bc : P in BC network
    """
    all_nodes = sorted(set(part_aff) | set(part_bc))
    rows = []

    for node in all_nodes:
        c_aff = part_aff.get(node)
        c_bc  = part_bc.get(node)
        stable = STABLE_LABEL if c_aff == c_bc else MOVER_LABEL

        # -- affiliation network metrics --
        deg_aff = G_aff.degree(node, weight="weight") if node in G_aff else None
        intra_aff, cross_aff, P_aff = _community_edge_stats(node, G_aff, part_aff)

        # -- BC network metrics --
        deg_bc  = G_bc.degree(node, weight="weight") if node in G_bc else None
        intra_bc, cross_bc, P_bc = _community_edge_stats(node, G_bc, part_bc)

        rows.append({
            "client_name":      node,
            "community_aff":    c_aff,
            "community_bc":     c_bc,
            "stability":        stable,
            "degree_aff":       deg_aff,
            "degree_bc":        deg_bc,
            "intra_edges_aff":  intra_aff,
            "cross_edges_aff":  cross_aff,
            "participation_aff": P_aff,
            "intra_edges_bc":   intra_bc,
            "cross_edges_bc":   cross_bc,
            "participation_bc":  P_bc,
        })

    return pd.DataFrame(rows)


def _community_edge_stats(node, G, partition):
    """
    For a node in graph G with community labels in partition, compute:
      - intra: sum of weights to same-community neighbors
      - cross: sum of weights to different-community neighbors
      - P: participation coefficient = 1 - Σ_c (k_ic / k_i)^2
    Returns (intra_weight, cross_weight, P) or (0, 0, None) if node not in G.
    """
    if node not in G or node not in partition:
        return 0, 0, None

    my_comm = partition[node]
    total_w = G.degree(node, weight="weight")
    if total_w == 0:
        return 0, 0, 0.0

    # Weight to each community
    comm_weights = {}
    for nbr, data in G[node].items():
        nbr_comm = partition.get(nbr)
        if nbr_comm is None:
            continue
        w = data.get("weight", 1.0)
        comm_weights[nbr_comm] = comm_weights.get(nbr_comm, 0.0) + w

    intra = comm_weights.get(my_comm, 0.0)
    cross = total_w - intra
    P = 1.0 - sum((w / total_w) ** 2 for w in comm_weights.values())
    return round(intra, 4), round(cross, 4), round(P, 4)


# -- Main --

def main():
    print("Building affiliation and BC similarity graphs…")
    G_aff = build_affiliation_graph()
    G_bc  = build_bc_graph()

    print(f"\nAffiliation graph: {G_aff.number_of_nodes()} nodes, "
          f"{G_aff.number_of_edges()} edges")
    print(f"BC similarity graph: {G_bc.number_of_nodes()} nodes, "
          f"{G_bc.number_of_edges()} edges")

    # -- Detect communities --
    print(f"\n-- Leiden community detection (γ={LEIDEN_RESOLUTION}) --")

    part_aff, Q_aff, summ_aff = detect_communities(
        G_aff, resolution=LEIDEN_RESOLUTION, seed=SEED
    )
    print(f"\nAffiliation network  Q={Q_aff:.4f}  "
          f"{len(summ_aff)} communities")
    for cid, size in community_sizes(summ_aff):
        print(f"  Community {cid:2d}: {size:3d} firms")

    part_bc, Q_bc, summ_bc = detect_communities(
        G_bc, resolution=LEIDEN_RESOLUTION, seed=SEED
    )
    print(f"\nBC similarity network  Q={Q_bc:.4f}  "
          f"{len(summ_bc)} communities")
    for cid, size in community_sizes(summ_bc):
        print(f"  Community {cid:2d}: {size:3d} firms")

    # -- Global agreement --
    nmi, ari, n_shared = compute_global_agreement(part_aff, part_bc)
    print(f"\n-- Global Partition Agreement ({n_shared} shared firms) --")
    print(f"  NMI = {nmi:.4f}  (0=random, 1=identical)")
    print(f"  ARI = {ari:.4f}  (0=random, 1=identical, <0=worse than random)")

    # -- Confusion matrix --
    # Align BC community IDs to affiliation IDs before building the matrix
    part_bc_aligned = align_community_ids(part_aff, part_bc)
    conf = confusion_matrix_df(part_aff, part_bc_aligned,
                               label_a="community_aff", label_b="community_bc")
    print(f"\n-- Community Confusion Matrix (rows=affiliation, cols=BC) --")
    print(conf.to_string())

    conf_path = DATA_DIR / "community_confusion.csv"
    conf.to_csv(conf_path)
    print(f"\n  Confusion matrix exported → {conf_path}")

    # -- Firm-level stability table --
    stability = firm_stability_table(part_aff, part_bc_aligned, G_aff, G_bc)

    n_stable = (stability["stability"] == STABLE_LABEL).sum()
    n_cross  = (stability["stability"] == MOVER_LABEL).sum()
    print(f"\n-- Firm-Level Stability --")
    print(f"  Stable (same community in both networks): {n_stable} firms")
    print(f"  Cross-community movers:                   {n_cross} firms")

    # Print movers sorted by participation coefficient in BC network
    movers = (stability[stability["stability"] == MOVER_LABEL]
              .sort_values("participation_bc", ascending=False))
    print(f"\n  Cross-community movers (sorted by BC participation coeff):")
    print(f"  {'Firm':<44} {'Aff':>5} {'BC':>5} "
          f"{'P_aff':>7} {'P_bc':>7}")
    print(f"  {'-'*44} {'-'*5} {'-'*5} {'-'*7} {'-'*7}")
    for _, row in movers.iterrows():
        print(f"  {row['client_name']:<44} "
              f"{str(row['community_aff']):>5} {str(row['community_bc']):>5} "
              f"{str(row['participation_aff']):>7} {str(row['participation_bc']):>7}")

    # Print top stable firms by BC participation (provincial hubs)
    stable_df = (stability[stability["stability"] == STABLE_LABEL]
                 .sort_values("participation_bc", ascending=False))
    print(f"\n  Top 20 stable firms by BC participation coeff "
          f"(cross-industry within stable membership):")
    print(f"  {'Firm':<44} {'Comm':>5} {'P_bc':>7} {'Degree_bc':>10}")
    print(f"  {'-'*44} {'-'*5} {'-'*7} {'-'*10}")
    for _, row in stable_df.head(20).iterrows():
        print(f"  {row['client_name']:<44} "
              f"{str(row['community_aff']):>5} "
              f"{str(row['participation_bc']):>7} "
              f"{str(round(row['degree_bc'], 3) if row['degree_bc'] else '—'):>10}")

    # -- Export --
    out = DATA_DIR / "community_comparison.csv"
    stability.to_csv(out, index=False)
    print(f"\n  Full firm-level table exported → {out}")


if __name__ == "__main__":
    main()
