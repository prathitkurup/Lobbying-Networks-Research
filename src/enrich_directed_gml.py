"""
Enrich the RBO directed influence GML with additional node metrics.

New node attributes added
--------------------------
  num_bills              : unique bills lobbied per firm (post MAX_BILL_DF prevalence
                           filter, consistent with network construction pipeline)
  bill_aff_community     : Leiden community label from the bill affiliation network
                           (-1 sentinel for firms absent from that network)
  within_comm_net_str    : within-community net RBO strength; sum of RBO weights on
                           directed (balanced=0) out-edges to same-community peers
                           minus in-edges from same-community peers.
                           Balanced edges excluded — their canonical direction is
                           alphabetical and introduces no real directional signal
                           (consistent with global net_strength convention).
  within_comm_net_inf    : within-community net influence (count-based);
                           first-mover wins minus losses on directed (balanced=0)
                           edges where both endpoints share the same bill affiliation
                           community.

Inputs
------
  visualizations/gml/rbo_directed_influence.gml   (DiGraph to enrich)
  data/opensecrets_lda_reports.csv                (bill count source)
  data/communities/communities_affiliation.csv    (bill affiliation communities)

Output
------
  visualizations/gml/rbo_directed_influence.gml   (overwritten in-place)

See design_decisions.md §22 for methodology notes.
"""

import sys
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence

GML_PATH  = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"
COMM_PATH = DATA_DIR / "communities" / "communities_affiliation.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_graph(path):
    """Load directed GML; nodes keyed by firm name string (label attr)."""
    G = nx.read_gml(str(path), label="label")
    print(f"Loaded GML: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_num_bills(csv_path):
    """
    Count unique bills per firm after MAX_BILL_DF prevalence filter.

    Returns
    -------
    dict {firm_name: int}
    """
    df = load_bills_data(csv_path)
    df_dedup = df.drop_duplicates(subset=["fortune_name", "bill_number"])
    if MAX_BILL_DF is not None:
        df_dedup = filter_bills_by_prevalence(
            df_dedup, MAX_BILL_DF, unit_col="bill_number"
        )
    counts = df_dedup.groupby("fortune_name")["bill_number"].nunique().to_dict()
    print(
        f"  num_bills: {len(counts)} firms  |  "
        f"range [{min(counts.values())}, {max(counts.values())}]"
    )
    return counts


def load_bill_aff_communities(path):
    """
    Load bill affiliation Leiden community assignments.

    Returns
    -------
    dict {firm_name: community_id (int)}
    """
    df = pd.read_csv(path)
    mapping = dict(zip(df["fortune_name"], df["community_aff"].astype(int)))
    print(
        f"  bill_aff_community: {len(mapping)} firms  |  "
        f"{df['community_aff'].nunique()} communities"
    )
    return mapping


# ---------------------------------------------------------------------------
# Within-community metric computation
# ---------------------------------------------------------------------------

def compute_within_community_metrics(G, community_map):
    """
    Compute within-community net strength and net influence for every node,
    restricted to directed (balanced=0) edges where both endpoints belong to
    the same bill affiliation community.

    Within-community net strength
      = sum(weight) on balanced=0 out-edges to same-community peers
        - sum(weight) on balanced=0 in-edges from same-community peers

    Within-community net influence
      = (out_sf_wc + in_tf_wc) - (out_tf_wc + in_sf_wc)
      where, for edges involving node v and a same-community peer:
        out_sf / out_tf  source_firsts / target_firsts on balanced=0 out-edges
        in_sf  / in_tf   source_firsts / target_firsts on balanced=0 in-edges
      This mirrors the global net_influence calculation in build_graph().

    Parameters
    ----------
    G             : DiGraph loaded from rbo_directed_influence.gml
    community_map : {firm: community_id} from bill affiliation network

    Returns
    -------
    wc_net_str : {node: float}   within-community net RBO strength
    wc_net_inf : {node: int}     within-community net influence
    """
    wc_net_str = {}
    wc_net_inf = {}

    for node in G.nodes():
        c_v = community_map.get(node)

        dir_out_str  = 0.0
        dir_in_str   = 0.0
        out_sf_wc = out_tf_wc = 0
        in_sf_wc  = in_tf_wc  = 0

        for _, u, d in G.out_edges(node, data=True):
            if d.get("balanced", 1) == 1:
                continue                            # skip balanced edges
            if community_map.get(u) != c_v:
                continue                            # skip cross-community edges
            dir_out_str += d["weight"]
            out_sf_wc   += d["source_firsts"]
            out_tf_wc   += d["target_firsts"]

        for u, _, d in G.in_edges(node, data=True):
            if d.get("balanced", 1) == 1:
                continue
            if community_map.get(u) != c_v:
                continue
            dir_in_str += d["weight"]
            in_sf_wc   += d["source_firsts"]
            in_tf_wc   += d["target_firsts"]

        wc_net_str[node] = round(dir_out_str - dir_in_str, 4)
        wc_net_inf[node] = int((out_sf_wc + in_tf_wc) - (out_tf_wc + in_sf_wc))

    return wc_net_str, wc_net_inf


# ---------------------------------------------------------------------------
# Node annotation
# ---------------------------------------------------------------------------

def annotate_nodes(G, num_bills, comm_map, wc_net_str, wc_net_inf):
    """
    Write four new attributes onto G nodes in-place.
    Sentinel -1 is used for firms missing from the source datasets.
    """
    missing_bills = []
    missing_comm  = []

    for node in G.nodes():
        nb = num_bills.get(node)
        if nb is None:
            missing_bills.append(node)
            nb = -1
        G.nodes[node]["num_bills"] = int(nb)

        ca = comm_map.get(node)
        if ca is None:
            missing_comm.append(node)
            ca = -1
        G.nodes[node]["bill_aff_community"] = int(ca)

        G.nodes[node]["within_comm_net_str"] = float(wc_net_str.get(node, 0.0))
        G.nodes[node]["within_comm_net_inf"] = int(wc_net_inf.get(node, 0))

    if missing_bills:
        print(
            f"  Warning: {len(missing_bills)} node(s) missing num_bills → -1 sentinel: "
            f"{missing_bills[:5]}{'...' if len(missing_bills) > 5 else ''}"
        )
    if missing_comm:
        print(
            f"  Warning: {len(missing_comm)} node(s) missing bill_aff_community → -1 sentinel: "
            f"{missing_comm[:5]}{'...' if len(missing_comm) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(G):
    """Print top/bottom nodes by within_comm_net_inf as a sanity check."""
    nodes_sorted = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n]["within_comm_net_inf"],
        reverse=True,
    )
    header = f"  {'Firm':<42} {'Comm':>5} {'Bills':>6} {'WC Net Str':>12} {'WC Net Inf':>12}"
    print(f"\n  Top 8 by within_comm_net_inf:")
    print(header)
    for n in nodes_sorted[:8]:
        a = G.nodes[n]
        print(
            f"  {n:<42} {a['bill_aff_community']:>5} {a['num_bills']:>6} "
            f"{a['within_comm_net_str']:>12.4f} {a['within_comm_net_inf']:>12}"
        )
    print(f"\n  Bottom 5 by within_comm_net_inf:")
    print(header)
    for n in nodes_sorted[-5:]:
        a = G.nodes[n]
        print(
            f"  {n:<42} {a['bill_aff_community']:>5} {a['num_bills']:>6} "
            f"{a['within_comm_net_str']:>12.4f} {a['within_comm_net_inf']:>12}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading graph...")
    G = load_graph(GML_PATH)

    print("Loading num_bills (post prevalence filter)...")
    num_bills = load_num_bills(DATA_DIR / "opensecrets_lda_reports.csv")

    print("Loading bill affiliation communities...")
    comm_map = load_bill_aff_communities(COMM_PATH)

    print("Computing within-community metrics...")
    wc_net_str, wc_net_inf = compute_within_community_metrics(G, comm_map)

    print("Annotating nodes...")
    annotate_nodes(G, num_bills, comm_map, wc_net_str, wc_net_inf)

    print(f"\nWriting enriched GML -> {GML_PATH}")
    nx.write_gml(G, str(GML_PATH))
    print(f"  Done: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print_summary(G)


if __name__ == "__main__":
    main()
