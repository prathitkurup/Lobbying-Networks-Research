"""
Compare Leiden community partitions across four co-lobbying networks:
composite, affiliation, cosine, and RBO. Computes pairwise NMI/ARI and
a per-firm consensus stability classification.

Outputs:
  data/community_comparison_composite.csv  - per-firm assignments + stability label
  data/nmi_ari_matrix.csv                  - NMI and ARI for all pairwise comparisons

Run: python composite_community_comparison.py
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

LEIDEN_RESOLUTION = 1.0
SEED              = 42

NETWORK_LABELS = ["composite", "affiliation", "cosine", "rbo"]


def align_community_ids(part_a, part_b):
    """Re-label community IDs in part_b to maximally overlap with part_a using Hungarian assignment."""
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
    nmi = normalized_mutual_info_score(labels_a, labels_b, average_method="arithmetic")
    ari = adjusted_rand_score(labels_a, labels_b)
    return nmi, ari, len(shared)


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
            nmi_mat.loc[la, lb] = round(nmi, 4)
            nmi_mat.loc[lb, la] = round(nmi, 4)
            ari_mat.loc[la, lb] = round(ari, 4)
            ari_mat.loc[lb, la] = round(ari, 4)
    return nmi_mat, ari_mat


def consensus_stability_table(partitions, aligned_partitions):
    """
    Build per-firm table with community assignments and stability label.

    Stability labels:
      fully_stable          - same aligned community in all four networks
      partially_stable      - matches composite in >= 1 other network
      composite_divergent   - composite places firm differently than all three others
      absent_from_composite - firm not in the composite network
    """
    all_nodes = sorted(set().union(*[set(p.keys()) for p in partitions.values()]))
    rows = []
    for node in all_nodes:
        row = {"fortune_name": node}
        comms = {}
        for label in NETWORK_LABELS:
            cid = aligned_partitions.get(label, {}).get(node)
            row[f"community_{label}"] = cid
            if cid is not None:
                comms[label] = cid

        comp_comm = comms.get("composite")
        if comp_comm is None:
            row["consensus_count"] = 0
            row["stability_label"] = "absent_from_composite"
        else:
            agree = sum(
                1 for lbl in ["affiliation", "cosine", "rbo"]
                if comms.get(lbl) == comp_comm
            )
            row["consensus_count"] = agree
            if agree == 3:
                row["stability_label"] = "fully_stable"
            elif agree >= 1:
                row["stability_label"] = "partially_stable"
            else:
                row["stability_label"] = "composite_divergent"
        rows.append(row)

    return pd.DataFrame(rows)


def confusion_matrix_df(part_a, part_b, label_a="a", label_b="b"):
    """Return a pivot-table confusion matrix between two partitions."""
    shared = set(part_a) & set(part_b)
    records = [(part_a[n], part_b[n]) for n in shared]
    df = pd.DataFrame(records, columns=[label_a, label_b])
    return df.pivot_table(index=label_a, columns=label_b,
                          aggfunc="size", fill_value=0)


def main():
    print("Loading pre-computed edge CSVs...\n")

    affil_path  = DATA_DIR / "affiliation_edges.csv"
    cosine_path = DATA_DIR / "cosine_edges.csv"
    rbo_path    = DATA_DIR / "rbo_edges.csv"
    comp_path   = DATA_DIR / "composite_edges.csv"

    for p in (affil_path, cosine_path, rbo_path, comp_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found. Run the corresponding network script first."
            )

    affil_edges  = pd.read_csv(affil_path)
    cosine_edges = pd.read_csv(cosine_path)
    rbo_edges    = pd.read_csv(rbo_path)
    comp_edges   = pd.read_csv(comp_path)

    G_affil  = build_graph(affil_edges[["source", "target", "weight"]])
    G_cosine = build_graph(cosine_edges[["source", "target", "weight"]])
    G_rbo    = build_graph(rbo_edges[["source", "target", "weight"]])
    G_comp   = build_graph_with_attrs(comp_edges, weight_col="weight")

    graphs = {
        "composite":   G_comp,
        "affiliation": G_affil,
        "cosine":      G_cosine,
        "rbo":         G_rbo,
    }

    print("-- Network sizes --")
    for lbl, G in graphs.items():
        print(f"  {lbl:<14} {G.number_of_nodes():>4} nodes  "
              f"{G.number_of_edges():>6} edges")

    print(f"\n-- Leiden (resolution={LEIDEN_RESOLUTION}) community detection --")
    partitions = {}
    for lbl, G in graphs.items():
        part, Q, summ = detect_communities(G, resolution=LEIDEN_RESOLUTION, seed=SEED)
        partitions[lbl] = part
        n_comms = len(set(part.values()))
        print(f"\n  {lbl.upper():<14} Q={Q:.4f}  {n_comms} communities")
        for cid, size in community_sizes(summ):
            print(f"    Community {cid:2d}: {size:3d} firms")

    print(f"\n-- Pairwise Partition Agreement --")
    nmi_mat, ari_mat = pairwise_agreement_matrix(partitions)

    print("\n  NMI matrix (1=identical, 0=independent):")
    print(nmi_mat.to_string())
    print("\n  ARI matrix (1=identical, 0=random, <0=worse than random):")
    print(ari_mat.to_string())

    nmi_mat.index.name = "network"
    ari_mat.index.name = "network"
    nmi_export = nmi_mat.copy(); nmi_export["metric"] = "NMI"
    ari_export = ari_mat.copy(); ari_export["metric"] = "ARI"
    pd.concat([nmi_export, ari_export]).to_csv(DATA_DIR / "nmi_ari_matrix.csv")
    print(f"\nNMI/ARI matrix -> {DATA_DIR / 'nmi_ari_matrix.csv'}")

    print(f"\n-- Composite vs Individual Networks --")
    for lbl in ["affiliation", "cosine", "rbo"]:
        nmi_val = nmi_mat.loc["composite", lbl]
        ari_val = ari_mat.loc["composite", lbl]
        _, _, n_shared = compute_global_agreement(partitions["composite"],
                                                  partitions[lbl])
        print(f"  composite vs {lbl:<12}  NMI={nmi_val:.4f}  "
              f"ARI={ari_val:.4f}  (n={n_shared})")

    print(f"\n  NMI > 0.70: strong structural alignment between partitions")
    print(f"  NMI 0.40-0.70: moderate overlap; some community reorganization")
    print(f"  NMI < 0.40: meaningful divergence; metrics capture distinct structure")

    aligned = {"composite": partitions["composite"]}
    for lbl in ["affiliation", "cosine", "rbo"]:
        aligned[lbl] = align_community_ids(partitions["composite"], partitions[lbl])

    stability = consensus_stability_table(partitions, aligned)
    counts = stability["stability_label"].value_counts()
    print(f"\n-- Firm-Level Consensus Stability --")
    for label, count in counts.items():
        print(f"  {label:<30}: {count:>4} firms")

    divergent = stability[stability["stability_label"] == "composite_divergent"].sort_values("fortune_name")
    if not divergent.empty:
        print(f"\n  Composite-divergent firms ({len(divergent)}):")
        print(f"  {'Firm':<44} {'Comp':>5} {'Aff':>5} {'Cos':>5} {'RBO':>5}")
        print(f"  {'-'*44} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
        for _, row in divergent.iterrows():
            print(f"  {row['fortune_name']:<44} "
                  f"{str(row['community_composite']):>5} "
                  f"{str(row['community_affiliation']):>5} "
                  f"{str(row['community_cosine']):>5} "
                  f"{str(row['community_rbo']):>5}")

    stable = stability[stability["stability_label"] == "fully_stable"].sort_values("fortune_name")
    print(f"\n  Fully stable firms ({len(stable)}) - identical community across all 4 networks:")
    print(f"  {'Firm':<44} {'Community':>10}")
    for _, row in stable.head(30).iterrows():
        print(f"  {row['fortune_name']:<44} {str(row['community_composite']):>10}")
    if len(stable) > 30:
        print(f"  ... ({len(stable)-30} more)")

    for lbl in ["affiliation", "cosine", "rbo"]:
        conf = confusion_matrix_df(aligned["composite"], aligned[lbl],
                                   label_a="composite", label_b=lbl)
        print(f"\n-- Confusion Matrix: composite vs {lbl} --")
        print(conf.to_string())

    out_path = DATA_DIR / "community_comparison_composite.csv"
    stability.to_csv(out_path, index=False)
    print(f"\nFull firm-level table -> {out_path}")


if __name__ == "__main__":
    main()
