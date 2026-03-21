"""
Centrality utilities — two tiers.

Tier 1 (global): compute_centralities() — standard NetworkX metrics on the
full graph. Identifies overall network hubs regardless of community structure.

Tier 2 (community-based): compute_community_centralities() — three measures
that require a community partition:

  1. within_community_eigenvector
       Eigenvector centrality run on each community subgraph independently.
       Superior to z-score (normalized weighted degree) because eigenvector
       captures the recursive "important neighbors" structure: a firm is central
       not just because it has many strong connections, but because those
       connections are themselves to highly connected firms. This identifies
       the true industry leaders — the firms that are well-connected to other
       well-connected firms within the same lobbying coalition.

  2. z_score  (Guimerà-Amaral within-community degree z-score)
       z_i = (κ_is - mean(κ_s)) / std(κ_s)
       where κ_is = sum of edge weights from i to same-community nodes.
       The original Guimerà & Amaral (2005, Nature) formulation. Included
       alongside eigenvector for methodological comparison.

  3. participation_coefficient  (Guimerà-Amaral P)
       P_i = 1 - Σ_c [(κ_ic / κ_i)²]
       where κ_ic = weight of edges from i to community c.
       P = 0: all connections within own community (pure industry player).
       P -> 1: connections spread evenly across all communities (cross-industry).
       This is the key measure for identifying cross-industry political
       entrepreneurs — firms that bridge across lobbying coalitions.

  4. global_pagerank
       PageRank on the full graph. Identifies overall network hubs.
       Comparison to within-community eigenvector reveals whether a firm's
       global prominence comes from within-industry dominance or cross-industry
       bridging.

  Guimerà-Amaral node role classification combines z and P:
    provincial_hub:   z ≥ 2.5, P < 0.30  (dominant in community, stays in lane)
    connector_hub:    z ≥ 2.5, 0.30 ≤ P < 0.75  (dominant AND cross-industry)
    kinless_hub:      z ≥ 2.5, P ≥ 0.75  (highly cross-industry hub)
    ultra_peripheral: z < 2.5, P < 0.05  (almost entirely within community)
    peripheral:       z < 2.5, 0.05 ≤ P < 0.625
    non_hub_connector:z < 2.5, 0.625 ≤ P < 0.80  (bridges without dominance)
    kinless:          z < 2.5, P ≥ 0.80

Reference: Guimerà, R. & Amaral, L.A.N. (2005). Functional cartography of
complex metabolic networks. Nature, 433, 895-900.
"""

import numpy as np
import pandas as pd
import networkx as nx


# -- Tier 1: global centrality --

def compute_katz_centrality(G, weight_attr="weight", normalized=True, max_iter=2000):
    """
    Katz-Bonacich centrality for weighted graphs.

    C_katz(i) = alpha * sum_j A_ij * C_katz(j) + beta

    alpha must be < 1/lambda_max (spectral radius of the weighted adjacency
    matrix) to guarantee convergence.  We auto-set alpha = 0.85 / rho, where
    rho is the spectral radius computed via numpy's eigenvalue solver.  If rho
    is zero (empty or all-zero-weight graph), we fall back to all-zeros.

    The Katz penalty on longer paths (exponential decay in path length) is
    particularly informative for lobbying networks because it distinguishes
    firms that are influential through direct strong ties from firms whose
    prominence depends on being embedded in a dense multi-hop neighbourhood.
    Contrast with PageRank (which also uses path-length decay but normalises
    by out-degree) and within-community eigenvector (community-scoped only).

    Parameters
    ----------
    G          : NetworkX Graph with weighted edges.
    weight_attr: Edge attribute name for weights.
    normalized : Whether to L2-normalise the final vector.
    max_iter   : Maximum power-iteration steps.

    Returns
    -------
    dict : {node: katz_centrality_value}

    Reference
    ---------
    Katz, L. (1953). A new status index derived from sociometric analysis.
    Psychometrika, 18(1), 39-43.
    Bonacich, P. (1987). Power and centrality: A family of measures. American
    Journal of Sociology, 92(5), 1170-1182.
    """
    if G.number_of_nodes() == 0:
        return {}

    A = nx.to_numpy_array(G, weight=weight_attr)
    try:
        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        spectral_radius = 0.0

    if spectral_radius == 0.0:
        return {n: 0.0 for n in G.nodes()}

    alpha = 0.85 / spectral_radius

    try:
        katz = nx.katz_centrality(
            G, alpha=alpha, beta=1.0, weight=weight_attr,
            max_iter=max_iter, normalized=normalized,
        )
    except nx.PowerIterationFailedConvergence:
        # More conservative alpha if convergence fails at 0.85/rho
        alpha = 0.50 / spectral_radius
        try:
            katz = nx.katz_centrality(
                G, alpha=alpha, beta=1.0, weight=weight_attr,
                max_iter=max_iter * 2, normalized=normalized,
            )
        except nx.PowerIterationFailedConvergence:
            # Last-resort fallback: weighted degree (same units, no path penalty)
            deg = dict(G.degree(weight=weight_attr))
            max_d = max(deg.values()) or 1.0
            katz = {n: v / max_d for n, v in deg.items()}

    return katz


def compute_centralities(G):
    return {
        "degree":      nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "closeness":   nx.closeness_centrality(G, distance="weight"),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight",
                                                  max_iter=1000),
    }


def print_top_centralities(centralities, k=10):
    for name, values in centralities.items():
        print(f"\nTop {k} by {name.upper()}:")
        for node, val in sorted(values.items(), key=lambda x: x[1],
                                 reverse=True)[:k]:
            print(f"  {node}: {val:.4f}")


# -- Tier 2 helpers --

def _community_map(partition):
    """Return {community_id: [node, ...]} from {node: community_id}."""
    comm_map = {}
    for node, cid in partition.items():
        comm_map.setdefault(cid, []).append(node)
    return comm_map


def compute_within_community_eigenvector(G, partition, weight_attr="weight"):
    """
    Eigenvector centrality computed on each community subgraph.
    Falls back to weighted degree centrality for communities with < 3 nodes
    or when the power iteration doesn't converge.
    Returns {node: centrality_value}.
    """
    comm_map = _community_map(partition)
    ec = {}
    for cid, members in comm_map.items():
        sub = G.subgraph(members)
        if len(members) < 3:
            # Eigenvector is ill-defined for very small subgraphs.
            deg = nx.degree_centrality(sub)
            ec.update(deg)
        else:
            try:
                ev = nx.eigenvector_centrality(sub, weight=weight_attr,
                                               max_iter=2000)
                ec.update(ev)
            except nx.PowerIterationFailedConvergence:
                # Fall back to weighted degree for this community.
                total_w = {n: sub.degree(n, weight=weight_attr) for n in sub}
                max_w = max(total_w.values()) or 1
                ec.update({n: v / max_w for n, v in total_w.items()})
    return ec


def compute_within_community_zscore(G, partition, weight_attr="weight"):
    """
    Guimerà-Amaral z-score: normalized within-community weighted degree.
    z_i = (κ_is - mean(κ_s)) / std(κ_s)
    where κ_is = sum of edge weights from i to same-community nodes.
    Returns {node: z_score}.
    """
    comm_map = _community_map(partition)

    # Compute κ_ic (within-community weighted degree) for every node.
    kic = {}
    for node in G.nodes():
        my_comm = partition.get(node)
        if my_comm is None:
            kic[node] = 0.0
            continue
        kic[node] = sum(
            d.get(weight_attr, 1.0)
            for nbr, d in G[node].items()
            if partition.get(nbr) == my_comm
        )

    # Compute z per community.
    z = {}
    for cid, members in comm_map.items():
        vals = [kic[m] for m in members if m in kic]
        mean_k = np.mean(vals) if vals else 0.0
        std_k  = np.std(vals,  ddof=0) if len(vals) > 1 else 0.0
        for m in members:
            if std_k > 0:
                z[m] = (kic.get(m, 0.0) - mean_k) / std_k
            else:
                z[m] = 0.0
    return z


def compute_participation_coefficient(G, partition, weight_attr="weight"):
    """
    Guimerà-Amaral participation coefficient.
    P_i = 1 - Σ_c [(κ_ic / κ_i)²]
    where κ_ic = total edge weight from i to community c,
          κ_i  = total weighted degree of i.
    P = 0: pure within-community player.  P -> 1: kinless cross-industry bridger.
    Returns {node: P_value}.
    """
    P = {}
    for node in G.nodes():
        if node not in partition:
            P[node] = None
            continue
        ki = G.degree(node, weight=weight_attr)
        if ki == 0:
            P[node] = 0.0
            continue
        comm_weights = {}
        for nbr, data in G[node].items():
            c = partition.get(nbr)
            if c is None:
                continue
            comm_weights[c] = comm_weights.get(c, 0.0) + data.get(weight_attr, 1.0)
        P[node] = 1.0 - sum((w / ki) ** 2 for w in comm_weights.values())
    return P


def classify_ga_role(z, P):
    """
    Guimerà-Amaral (2005) node role based on z-score and participation coeff.
    The connector_hub and non_hub_connector roles identify the cross-industry
    political entrepreneurs of greatest theoretical interest.
    """
    if P is None:
        return "unknown"
    if z >= 2.5:
        if P < 0.30:  return "provincial_hub"
        if P < 0.75:  return "connector_hub"
        return "kinless_hub"
    else:
        if P < 0.05:  return "ultra_peripheral"
        if P < 0.625: return "peripheral"
        if P < 0.80:  return "non_hub_connector"
        return "kinless"


# -- Tier 2: combined community centrality --

def compute_community_centralities(G, partition, weight_attr="weight"):
    """
    Compute all centrality tiers and return a DataFrame.

    Columns
    -------
    firm                    : node name
    community               : Leiden community ID
    within_comm_eigenvector : eigenvector centrality within community subgraph
    z_score                 : Guimerà-Amaral within-community degree z-score
    participation_coeff     : Guimerà-Amaral P (cross-community bridging)
    global_pagerank         : PageRank on full graph
    katz_centrality         : Katz-Bonacich centrality (alpha auto-calibrated to
                              0.85 / spectral_radius; captures influence through
                              all path lengths with exponential decay)
    ga_role                 : Guimerà-Amaral node role label
    """
    ec   = compute_within_community_eigenvector(G, partition, weight_attr)
    z    = compute_within_community_zscore(G, partition, weight_attr)
    P    = compute_participation_coefficient(G, partition, weight_attr)
    pr   = nx.pagerank(G, weight=weight_attr)
    katz = compute_katz_centrality(G, weight_attr=weight_attr)

    rows = []
    for node in sorted(G.nodes()):
        p_val = P.get(node)
        z_val = z.get(node, 0.0)
        rows.append({
            "firm":                    node,
            "community":               partition.get(node),
            "within_comm_eigenvector": round(ec.get(node, 0.0), 6),
            "z_score":                 round(z_val, 4),
            "participation_coeff":     round(p_val, 4) if p_val is not None else None,
            "global_pagerank":         round(pr.get(node, 0.0), 6),
            "katz_centrality":         round(katz.get(node, 0.0), 6),
            "ga_role":                 classify_ga_role(z_val, p_val),
        })

    return (pd.DataFrame(rows)
              .sort_values("within_comm_eigenvector", ascending=False)
              .reset_index(drop=True))


def print_community_centralities(cent_df, k=10):
    """
    Print a structured summary of all centrality tiers:
      — top-k by within-community eigenvector (industry leaders)
      — top-k by participation coefficient (cross-industry connectors)
      — top-k by global PageRank (overall network hubs)
      — top-k by Katz-Bonacich centrality (path-penalised influence)
      — Guimerà-Amaral role distribution
    """
    print("\n-- Community Centrality: Top Industry Leaders "
          "(within-community eigenvector) --")
    top_ec = cent_df.nlargest(k, "within_comm_eigenvector")
    print(f"  {'Firm':<42} {'Comm':>5}  {'EC':>8}  {'GA role'}")
    for _, r in top_ec.iterrows():
        print(f"  {r['firm']:<42} {str(r['community']):>5}  "
              f"{r['within_comm_eigenvector']:>8.4f}  {r['ga_role']}")

    print(f"\n-- Community Centrality: Top Cross-Industry Connectors "
          "(participation coeff) --")
    top_P = cent_df.dropna(subset=["participation_coeff"]).nlargest(
        k, "participation_coeff"
    )
    print(f"  {'Firm':<42} {'Comm':>5}  {'P':>8}  {'GA role'}")
    for _, r in top_P.iterrows():
        print(f"  {r['firm']:<42} {str(r['community']):>5}  "
              f"{r['participation_coeff']:>8.4f}  {r['ga_role']}")

    print(f"\n-- Community Centrality: Top Global Hubs (PageRank) --")
    top_pr = cent_df.nlargest(k, "global_pagerank")
    print(f"  {'Firm':<42} {'Comm':>5}  {'PR':>8}  {'GA role'}")
    for _, r in top_pr.iterrows():
        print(f"  {r['firm']:<42} {str(r['community']):>5}  "
              f"{r['global_pagerank']:>8.5f}  {r['ga_role']}")

    print(f"\n-- Community Centrality: Top Katz-Bonacich Hubs --")
    if "katz_centrality" in cent_df.columns:
        top_katz = cent_df.nlargest(k, "katz_centrality")
        print(f"  {'Firm':<42} {'Comm':>5}  {'Katz':>8}  {'GA role'}")
        for _, r in top_katz.iterrows():
            print(f"  {r['firm']:<42} {str(r['community']):>5}  "
                  f"{r['katz_centrality']:>8.5f}  {r['ga_role']}")
    else:
        print("  (katz_centrality not computed for this network)")

    print(f"\n-- Guimerà-Amaral Role Distribution --")
    role_counts = cent_df["ga_role"].value_counts()
    for role, count in role_counts.items():
        print(f"  {role:<22}: {count:>4}")
