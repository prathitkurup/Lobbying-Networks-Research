from pathlib import Path
import pandas as pd
import networkx as nx


def build_adjacency_matrix(edge_df):
    """Symmetric weighted adjacency matrix from an edge list."""
    forward  = edge_df[["source", "target", "weight"]]
    backward = edge_df.rename(columns={"source": "target", "target": "source"})[
        ["source", "target", "weight"]
    ]
    return pd.concat([forward, backward]).pivot_table(
        index="source", columns="target", values="weight", fill_value=0
    )


def build_graph(edge_df):
    """Build a weighted undirected NetworkX graph from an edge list."""
    return nx.from_pandas_edgelist(
        edge_df, source="source", target="target", edge_attr="weight"
    )


def write_gml_with_communities(G, partition, gml_path, node_attrs=None):
    """
    Write a GML file with community membership and optional analytics stored
    as node attributes for direct Gephi import.

    Parameters
    ----------
    G          : NetworkX graph (modified in-place with node attributes).
    partition  : {node: community_id} from Leiden.
    gml_path   : Output path for the GML file.
    node_attrs : Optional dict-of-dicts: {attr_name: {node: value, ...}}.
                 Supported attrs (written if present):
                   'within_comm_eigenvector', 'z_score',
                   'participation_coeff', 'global_pagerank',
                   'ga_role', 'kcore'
                 All numeric values are rounded to 6 decimal places.
                 String values (ga_role) are written as-is.
                 Gephi reads all node attributes as importable columns on load.

    Notes
    -----
    GML only supports int, float, and str node-attribute types. None values
    are converted to -1.0 (numeric) or 'unknown' (string) so the file remains
    valid. GML node labels are set to the node name string so Gephi displays
    firm names by default.
    """
    for node, cid in partition.items():
        if node in G:
            G.nodes[node]["community"] = int(cid)
            # GML requires 'label' to show names in Gephi's node table
            G.nodes[node]["label"] = str(node)

    if node_attrs:
        str_attrs = {"ga_role", "network_label"}
        for attr, mapping in node_attrs.items():
            is_str = attr in str_attrs
            for node, val in mapping.items():
                if node not in G:
                    continue
                if val is None or (not is_str and val != val):  # NaN guard
                    G.nodes[node][attr] = "unknown" if is_str else -1.0
                elif is_str:
                    G.nodes[node][attr] = str(val)
                else:
                    G.nodes[node][attr] = round(float(val), 6)

    Path(gml_path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(G, gml_path)
    n_attrs = (len(node_attrs) + 1) if node_attrs else 1  # +1 for community
    print(f"\nGML written ({n_attrs} node attrs) -> {gml_path}")


def _cent_df_to_attrs(cent_df):
    """
    Convert a compute_community_centralities() DataFrame into the
    node_attrs dict expected by write_gml_with_communities().
    Returns None if cent_df is None.

    Includes katz_centrality if present (added for composite network and
    back-filled for all networks via utils/centrality.py update).
    """
    if cent_df is None:
        return None
    firm_col = cent_df["firm"]
    attrs = {
        "within_comm_eigenvector": dict(zip(firm_col,
                                            cent_df["within_comm_eigenvector"])),
        "z_score":                 dict(zip(firm_col, cent_df["z_score"])),
        "participation_coeff":     dict(zip(firm_col,
                                            cent_df["participation_coeff"])),
        "global_pagerank":         dict(zip(firm_col, cent_df["global_pagerank"])),
        "ga_role":                 dict(zip(firm_col, cent_df["ga_role"])),
    }
    if "katz_centrality" in cent_df.columns:
        attrs["katz_centrality"] = dict(zip(firm_col, cent_df["katz_centrality"]))
    return attrs


def top_k_subgraph(G, k=20):
    """Subgraph of the top-k nodes by weighted degree (strength)."""
    strengths = {n: G.degree(n, weight="weight") for n in G.nodes()}
    top = sorted(strengths, key=strengths.get, reverse=True)[:k]
    return G.subgraph(top).copy()


def build_graph_with_attrs(edge_df, weight_col="weight"):
    """
    Build a weighted undirected NetworkX graph from a multi-column edge DataFrame.

    Unlike build_graph(), this function writes ALL numeric columns beyond
    'source' and 'target' as separate edge attributes so they are preserved
    in GML exports.  Useful for the composite network where cosine_weight,
    rbo_weight, ev_norm, and composite_weight should all be inspectable in Gephi.

    Parameters
    ----------
    edge_df    : DataFrame with at least [source, target, <weight_col>] columns.
                 Any additional numeric columns are added as edge attributes.
    weight_col : Name of the primary weight column (stored as 'weight' on the
                 edge for compatibility with NetworkX centrality and community
                 detection functions).

    Returns
    -------
    NetworkX Graph
    """
    attr_cols = [c for c in edge_df.columns if c not in ("source", "target")]
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        u, v = row["source"], row["target"]
        attrs = {col: float(row[col]) for col in attr_cols
                 if col != weight_col and not pd.isna(row[col])}
        attrs["weight"] = float(row[weight_col])
        G.add_edge(u, v, **attrs)
    return G


def edge_weight_stats(edge_df, label=""):
    """Print a concise summary of edge weight distribution."""
    w = edge_df["weight"]
    print(f"\n-- Edge weights{' (' + label + ')' if label else ''} --")
    print(f"  Edges: {len(w):,}  |  Mean: {w.mean():.3f}  |  Std: {w.std():.3f}")
    print(f"  Min: {w.min():.3f}  |  Max: {w.max():.3f}")
    print(f"  Percentiles — 25th: {w.quantile(.25):.3f}  "
          f"50th: {w.quantile(.50):.3f}  "
          f"75th: {w.quantile(.75):.3f}  "
          f"90th: {w.quantile(.90):.3f}")
