"""
[ARCHIVED] Compare Leiden community partitions across four co-lobbying networks.

Computes pairwise NMI/ARI and per-firm consensus stability classification.
Requires: composite_edges.csv, affiliation_edges.csv, cosine_edges.csv, rbo_edges.csv
in data/archive/network_edges/.
Outputs (archived): data/archive/community_comparison_composite.csv, nmi_ari_matrix.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, ".")
from config import DATA_DIR
from utils.network_building import build_graph, build_graph_with_attrs
from utils.community import detect_communities, print_community_summary, community_sizes

ARCHIVE           = DATA_DIR / "archive"
LEIDEN_RESOLUTION = 1.0
SEED              = 42
NETWORK_LABELS    = ["composite", "affiliation", "cosine", "rbo"]


def align_community_ids(part_a, part_b):
    """Re-label part_b community IDs to maximally overlap with part_a via Hungarian assignment."""
    ids_a = sorted(set(part_a.values()))
    ids_b = sorted(set(part_b.values()))
    overlap = np.zeros((len(ids_a), len(ids_b)), dtype=int)
    shared = set(part_a) & set(part_b)
    for node in shared:
        ia = ids_a.index(part_a[node])
        ib = ids_b.index(part_b[node])
        overlap[ia, ib] += 1
    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {ids_b[c]: ids_a[r] for r, c in zip(row_ind, col_ind)}
    next_id = max(ids_a) + 1
    for ib in ids_b:
        if ib not in mapping:
            mapping[ib] = next_id
            next_id += 1
    return {node: mapping[cid] for node, cid in part_b.items()}


def compute_global_agreement(part_a, part_b):
    """Return NMI, ARI, and shared node count between two partitions."""
    shared = sorted(set(part_a) & set(part_b))
    if len(shared) < 2:
        return 0.0, 0.0, len(shared)
    labels_a = [part_a[n] for n in shared]
    labels_b = [part_b[n] for n in shared]
    return (normalized_mutual_info_score(labels_a, labels_b, average_method="arithmetic"),
            adjusted_rand_score(labels_a, labels_b),
            len(shared))


def pairwise_agreement_matrix(partitions):
    """Build symmetric NMI and ARI matrices over all partition pairs."""
    labels = list(partitions.keys())
    nmi_mat = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    ari_mat = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i >= j:
                continue
            nmi, ari, _ = compute_global_agreement(partitions[la], partitions[lb])
            nmi_mat.loc[la, lb] = nmi_mat.loc[lb, la] = round(nmi, 4)
            ari_mat.loc[la, lb] = ari_mat.loc[lb, la] = round(ari, 4)
    return nmi_mat, ari_mat


def main():
    net_dir = ARCHIVE / "network_edges"
    composite_edges  = pd.read_csv(net_dir / "composite_edges.csv")
    affil_edges      = pd.read_csv(net_dir / "affiliation_edges.csv")
    cosine_edges     = pd.read_csv(net_dir / "cosine_edges.csv")
    rbo_edges        = pd.read_csv(net_dir / "rbo_edges.csv")

    graphs = {
        "composite":   build_graph_with_attrs(composite_edges, weight_col="weight"),
        "affiliation": build_graph(affil_edges[["source", "target", "weight"]]),
        "cosine":      build_graph(cosine_edges),
        "rbo":         build_graph(rbo_edges),
    }

    partitions = {}
    for name, G in graphs.items():
        res = 0.25 if name == "composite" else LEIDEN_RESOLUTION
        partition, Q, _ = detect_communities(G, resolution=res, seed=SEED)
        partitions[name] = partition
        print(f"{name}: Q={Q:.4f}  communities={len(set(partition.values()))}")

    nmi_mat, ari_mat = pairwise_agreement_matrix(partitions)
    combined = pd.concat([nmi_mat.add_prefix("nmi_"), ari_mat.add_prefix("ari_")], axis=1)
    combined.to_csv(ARCHIVE / "nmi_ari_matrix.csv")
    print(f"\nNMI/ARI matrix -> {ARCHIVE / 'nmi_ari_matrix.csv'}")


if __name__ == "__main__":
    main()
