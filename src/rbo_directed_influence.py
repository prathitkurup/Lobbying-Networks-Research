"""
Congress-wide directed RBO influence network — 116th Congress.

Builds a single aggregate network from all 8 quarters of lobbying data where:
  - Edge weight = RBO similarity (bill-priority ranking overlap, p=0.85)
  - Edge direction = temporal first-mover over shared top-30 bills
                    (global first quarter per (firm, bill); no double counting)
  - Balanced pairs = both A->B and B->A emitted with balanced=1 attribute
  - Node net_influence = net first-mover count (total firsts - total losses)
                         across ALL pairwise comparisons; for Gephi node sizing
  - Node color = #2ECC71 green (net > 0), #E74C3C red (net < 0), #95A5A6 gray (net == 0)

Output files:
  data/rbo_directed_influence.csv
  data/ranked_bill_lists.csv
  visualizations/gml/rbo_directed_influence.gml
  visualizations/png/rbo_directed_influence.png

See design_decisions.md §21 for full methodology.
"""

import sys
import itertools
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, ".")
from config import DATA_DIR, ROOT, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence
from utils.similarity import (
    aggregate_per_firm_bill, compute_zero_budget_fracs,
    build_ranked_lists, rbo_score,
)
from utils.visualization import plot_directed_circular

VIZ_GML_DIR = ROOT / "visualizations" / "gml"
VIZ_PNG_DIR = ROOT / "visualizations" / "png"

RBO_P     = 0.85   # Fortune 500-calibrated; see §18
TOP_BILLS = 30     # top bills per firm for ranking and RBO
TOP_K     = 20     # nodes included in PNG visualization
WRITE_CSV         = True
WRITE_RANKED_CSV  = True   # data/ranked_bill_lists.csv
WRITE_GML         = True
WRITE_PNG         = True


# -- Quarter assignment -------------------------------------------------------

def assign_quarters(df):
    """
    Add 'quarter' column: 2019 Q1-4 -> 1-4, 2020 Q1-4 -> 5-8.
    No-op if 'quarter' already present.
    """
    if "quarter" in df.columns:
        return df
    df = df.copy()
    base_q   = df["report_type"].str[1].astype(int)
    year_off = df["year"].map({2019: 0, 2020: 4})
    df["quarter"] = base_q + year_off
    return df


# -- Global first-quarter lookup ----------------------------------------------

def build_global_first_quarters(df):
    """
    Return {(firm, bill): min_quarter} using all 8 quarters.
    Captures the first time each firm lobbied each bill across the full congress.
    """
    return (
        df.groupby(["fortune_name", "bill_number"])["quarter"]
        .min()
        .to_dict()
    )


# -- Pairwise scoring ---------------------------------------------------------

def score_pair(firm_a, firm_b, shared_bills, bill_first):
    """
    First-mover tally for a firm pair over their shared top-30 bills.

    Compares global (congress-wide) first-quarter per (firm, bill).
    Each bill contributes at most 1 point — no double counting across quarters.

    Returns dict: a_firsts, b_firsts, tie_count, shared_bills.
    """
    a_firsts = b_firsts = ties = 0
    for bill in shared_bills:
        qa = bill_first.get((firm_a, bill))
        qb = bill_first.get((firm_b, bill))
        if qa is None or qb is None:
            continue   # defensive: bill missing from filing records
        if qa < qb:
            a_firsts += 1
        elif qb < qa:
            b_firsts += 1
        else:
            ties += 1
    return {
        "a_firsts":    a_firsts,
        "b_firsts":    b_firsts,
        "tie_count":   ties,
        "shared_bills": len(shared_bills),
    }


# -- Edge construction --------------------------------------------------------

def build_edges(ranked, bill_first, p=RBO_P):
    """
    Compute RBO similarity and temporal direction for all firm pairs
    that share at least one top-30 bill (RBO > 0).

    Direction rule:
      A_firsts > B_firsts → single directed A→B edge (balanced=0)
      B_firsts > A_firsts → single directed B→A edge (balanced=0)
      A_firsts == B_firsts → single canonical edge min(A,B)→max(A,B) (balanced=1);
                             canonical direction is alphabetical (arbitrary but consistent);
                             net_influence contribution is 0 for both nodes.

    Edge schema:
      source, target, weight (RBO score), source_firsts, target_firsts,
      tie_count, shared_bills, net_temporal (source_firsts − target_firsts),
      balanced (0/1)
    """
    firms   = sorted(ranked.keys())
    records = []

    for firm_a, firm_b in itertools.combinations(firms, 2):
        list_a = ranked[firm_a]
        list_b = ranked[firm_b]

        rbo_w = rbo_score(list_a, list_b, p=p)
        if rbo_w == 0.0:
            continue   # no shared top-30 bills; skip pair

        shared = set(list_a) & set(list_b)
        sc     = score_pair(firm_a, firm_b, shared, bill_first)
        a_f, b_f = sc["a_firsts"], sc["b_firsts"]
        balanced  = int(a_f == b_f)

        if balanced:
            # Single canonical edge: alphabetically min→max (avoids double-counting
            # weighted degree; direction is arbitrary but consistent across runs)
            src, tgt = (firm_a, firm_b) if firm_a < firm_b else (firm_b, firm_a)
            records.append({
                "source":        src,
                "target":        tgt,
                "weight":        round(rbo_w, 6),
                "source_firsts": a_f,
                "target_firsts": b_f,
                "tie_count":     sc["tie_count"],
                "shared_bills":  sc["shared_bills"],
                "net_temporal":  0,
                "balanced":      1,
            })
        else:
            # Directed: higher first-mover count wins direction
            src, tgt, sf, tf = (
                (firm_a, firm_b, a_f, b_f) if a_f > b_f
                else (firm_b, firm_a, b_f, a_f)
            )
            records.append({
                "source":        src,
                "target":        tgt,
                "weight":        round(rbo_w, 6),
                "source_firsts": sf,
                "target_firsts": tf,
                "tie_count":     sc["tie_count"],
                "shared_bills":  sc["shared_bills"],
                "net_temporal":  sf - tf,
                "balanced":      0,
            })

    cols = [
        "source", "target", "weight", "source_firsts", "target_firsts",
        "tie_count", "shared_bills", "net_temporal", "balanced",
    ]
    return pd.DataFrame(records) if records else pd.DataFrame(columns=cols)


# -- Graph construction -------------------------------------------------------

def build_graph(edges_df, ranked_firms=None):
    """
    Build DiGraph from edge records.

    Parameters
    ----------
    edges_df      : edge DataFrame from build_edges
    ranked_firms  : optional set of all firm names with ranked lists; when
                    provided, firms with no edges are added as isolated nodes
                    (net_influence=0, color='#95A5A6') so the full roster appears
                    in the GML and Gephi.

    Node attributes:
      net_influence = total pairwise first-mover wins - losses  [for Gephi sizing]
                    = (out_sf + in_tf) - (out_tf + in_sf)
                    where sf = source_firsts, tf = target_firsts on adjacent edges.
      total_firsts  = bills this node lobbied first across all pairings (always ≥ 0)
      total_losses  = bills this node lobbied second across all pairings (always ≥ 0)
      out_strength  = sum of RBO weights on outgoing edges  [graph-theoretic strength]
      in_strength   = sum of RBO weights on incoming edges  [graph-theoretic strength]
      net_strength  = out_strength(directed) - in_strength(directed), computed from
                      balanced=0 edges only; balanced edges excluded because their
                      canonical direction is alphabetical (arbitrary) and would
                      introduce a spurious RBO signal with no real directional meaning.
      color         = '#2ECC71' / '#E74C3C' / '#95A5A6'  (green / red / gray)
      label         = firm name
    """
    G = nx.DiGraph()

    # Add all ranked firms upfront so isolated firms appear as nodes
    if ranked_firms:
        for firm in ranked_firms:
            G.add_node(firm)

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source"], row["target"],
            weight=float(row["weight"]),
            source_firsts=int(row["source_firsts"]),
            target_firsts=int(row["target_firsts"]),
            tie_count=int(row["tie_count"]),
            shared_bills=int(row["shared_bills"]),
            net_temporal=int(row["net_temporal"]),
            balanced=int(row["balanced"]),
        )

    for node in G.nodes():
        # RBO-based strengths (for plot_directed_circular node sizing)
        # start=0.0 ensures float even when a node has no outgoing/incoming edges
        out_str = sum((d["weight"] for _, _, d in G.out_edges(node, data=True)), 0.0)
        in_str  = sum((d["weight"] for _, _, d in G.in_edges(node, data=True)), 0.0)

        # net_strength: RBO-weighted directional influence, directed edges only.
        # Balanced edges are excluded — their canonical direction is alphabetical
        # (arbitrary), so including them would produce a spurious net signal.
        dir_out_str = sum(
            (d["weight"] for _, _, d in G.out_edges(node, data=True) if d["balanced"] == 0),
            0.0,
        )
        dir_in_str = sum(
            (d["weight"] for _, _, d in G.in_edges(node, data=True) if d["balanced"] == 0),
            0.0,
        )

        # Count-based first-mover tallies
        # out edges: node is SOURCE  → node wins out_sf bills, loses out_tf bills
        out_sf = sum(d["source_firsts"] for _, _, d in G.out_edges(node, data=True))
        out_tf = sum(d["target_firsts"] for _, _, d in G.out_edges(node, data=True))
        # in edges: node is TARGET  → node wins in_tf bills, loses in_sf bills
        in_sf  = sum(d["source_firsts"] for _, _, d in G.in_edges(node, data=True))
        in_tf  = sum(d["target_firsts"] for _, _, d in G.in_edges(node, data=True))

        total_firsts = out_sf + in_tf
        total_losses = out_tf + in_sf
        net          = total_firsts - total_losses

        G.nodes[node]["out_strength"]  = round(out_str, 4)
        G.nodes[node]["in_strength"]   = round(in_str,  4)
        G.nodes[node]["net_strength"]  = round(dir_out_str - dir_in_str, 4)
        G.nodes[node]["total_firsts"]  = int(total_firsts)
        G.nodes[node]["total_losses"]  = int(total_losses)
        G.nodes[node]["net_influence"] = int(net)
        G.nodes[node]["label"]         = str(node)
        G.nodes[node]["color"] = (
            "#2ECC71" if net > 0 else ("#E74C3C" if net < 0 else "#95A5A6")
        )

    return G


# -- I/O helpers --------------------------------------------------------------

def export_ranked_lists(ranked, df_agg, output_path):
    """
    Export per-firm top-30 ranked bill lists to a long-format CSV.

    Columns: company, rank, bill_number, total_amount, budget_fraction.
    One row per (firm, rank); up to 30 rows per firm (~8,670 rows total).
    Sorted by company name then rank ascending.

    Parameters
    ----------
    ranked      : {firm: [bill, ...]} from build_ranked_lists
    df_agg      : post-filter DataFrame with fortune_name, bill_number,
                  amount_allocated, frac
    output_path : destination CSV path
    """
    lookup = (
        df_agg.set_index(["fortune_name", "bill_number"])
        [["amount_allocated", "frac"]]
        .to_dict("index")
    )
    rows = []
    for firm, bills in sorted(ranked.items()):
        for rank, bill in enumerate(bills, start=1):
            key    = (firm, bill)
            amount = lookup[key]["amount_allocated"] if key in lookup else 0.0
            frac   = lookup[key]["frac"]             if key in lookup else 0.0
            rows.append({
                "company":         firm,
                "rank":            rank,
                "bill_number":     bill,
                "total_amount":    round(amount, 2),
                "budget_fraction": round(frac, 6),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  Ranked lists CSV -> {Path(output_path).name}  "
          f"({len(ranked)} firms, {len(df_out)} rows)")
    return df_out


def write_gml(G, path):
    """Write directed GML for Gephi."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(G, path)
    print(f"  GML written ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges) -> {path}")


def print_stats(edges_df, G):
    """Print edge coverage, node tallies, and top agenda-setters / followers."""
    total    = len(edges_df)
    directed = (edges_df["balanced"] == 0).sum()
    balanced = (edges_df["balanced"] == 1).sum() // 2   # unique pairs
    print(f"\n  Total edges in DiGraph: {total:,}  "
          f"(directed: {directed:,}  |  balanced pairs: {balanced:,})")
    print(f"  Mean RBO weight:        {edges_df['weight'].mean():.4f}")
    if directed > 0:
        d_only = edges_df[edges_df["balanced"] == 0]
        print(f"  Mean net_temporal (directed): {d_only['net_temporal'].mean():.2f}")
        print(f"  Mean shared_bills  (directed): {d_only['shared_bills'].mean():.1f}")

    nodes_sorted = sorted(
        G.nodes(), key=lambda n: G.nodes[n]["net_influence"], reverse=True
    )
    print(f"\n  Top 8 agenda-setters (net_influence):")
    print(f"    {'Firm':<42} {'Firsts':>7}  {'Losses':>7}  {'Net':>7}")
    for n in nodes_sorted[:8]:
        a = G.nodes[n]
        sign = "+" if a["net_influence"] >= 0 else ""
        print(f"    {n:<42} {a['total_firsts']:>7}  {a['total_losses']:>7}  "
              f"{sign}{a['net_influence']:>6}")

    print(f"\n  Top 5 followers (lowest net_influence):")
    for n in nodes_sorted[-5:]:
        a = G.nodes[n]
        print(f"    {n:<42} {a['total_firsts']:>7}  {a['total_losses']:>7}  "
              f"{a['net_influence']:>7}")


# -- Main ---------------------------------------------------------------------

def main():
    # Load and assign quarters
    df_raw = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    df_raw = assign_quarters(df_raw)
    print(f"Loaded {len(df_raw):,} rows  |  "
          f"{df_raw['fortune_name'].nunique()} firms  |  "
          f"{df_raw['bill_number'].nunique()} bills")

    # Step 1: aggregate all quarters → fracs → prevalence filter
    df_agg = aggregate_per_firm_bill(df_raw)
    df_agg = compute_zero_budget_fracs(df_agg)
    if MAX_BILL_DF is not None:
        df_agg = filter_bills_by_prevalence(df_agg, MAX_BILL_DF, unit_col="bill_number")
    print(f"  After prevalence filter: {df_agg['bill_number'].nunique():,} bills")

    # Step 2: build top-30 ranked lists (by total spend across full congress)
    ranked = build_ranked_lists(df_agg, top_bills=TOP_BILLS)
    print(f"  Firms with ranked lists: {len(ranked):,}")

    if WRITE_RANKED_CSV:
        export_ranked_lists(ranked, df_agg, DATA_DIR / "ranked_bill_lists.csv")

    # Step 3: global first-quarter lookup (raw data; all quarters, all bills)
    bill_first = build_global_first_quarters(df_raw)
    print(f"  (firm, bill) first-quarter entries: {len(bill_first):,}")

    # Step 4: build edge list (RBO weight + temporal direction)
    print(f"\nBuilding edges (RBO p={RBO_P}, top_bills={TOP_BILLS}, "
          f"MAX_BILL_DF={MAX_BILL_DF})...")
    edges_df = build_edges(ranked, bill_first, p=RBO_P)
    print(f"  Done.")

    # Step 5: build graph and print stats
    G = build_graph(edges_df, ranked_firms=set(ranked.keys()))
    print_stats(edges_df, G)

    # Step 6: outputs
    if WRITE_CSV:
        csv_path = DATA_DIR / "rbo_directed_influence.csv"
        edges_df.to_csv(csv_path, index=False)
        print(f"\n  Edges CSV -> {csv_path.name}")

    if WRITE_GML:
        write_gml(G, str(VIZ_GML_DIR / "rbo_directed_influence.gml"))

    if WRITE_PNG:
        plot_directed_circular(
            G,
            title=f"Top {TOP_K} Congress-wide RBO Directed Influence Network",
            path=str(VIZ_PNG_DIR / "rbo_directed_influence.png"),
        )


if __name__ == "__main__":
    main()
