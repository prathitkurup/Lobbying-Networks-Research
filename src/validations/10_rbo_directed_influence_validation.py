"""
Validation suite for rbo_directed_influence.py outputs.

Checks:
  1.  Edge CSV schema — all required columns present, correct dtypes
  2.  RBO weight range — all weights in (0, 1]
  3.  Direction integrity — directed edges have source_firsts > target_firsts
  4.  net_temporal arithmetic — net_temporal == source_firsts - target_firsts
  5.  Balanced edge uniqueness — each firm pair appears at most once; canonical
      direction is min(A,B)→max(A,B) (source < target alphabetically)
  6.  Balanced net_temporal is zero — all balanced edges have net_temporal == 0
  7.  Node net_influence arithmetic — matches (out_sf + in_tf) - (out_tf + in_sf)
  8.  Node color consistency — green/red/gray matches sign of net_influence
  9.  Ranked-list CSV integrity — contiguous ranks, no gaps or duplicates per firm
  10. RBO spot-check — recompute RBO for 20 random pairs; values match CSV
  11. Node net_strength arithmetic — matches out_strength(directed) - in_strength(directed)

Run from src/ directory:
  python validations/10_rbo_directed_influence_validation.py
"""

import sys
import os
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT
from utils.similarity import rbo_score

EDGES_CSV  = DATA_DIR / "rbo_directed_influence.csv"
RANKED_CSV = DATA_DIR / "ranked_bill_lists.csv"
GML_PATH   = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"

PASS = "  PASS"
FAIL = "  FAIL"

EDGE_COLS = [
    "source", "target", "weight", "source_firsts", "target_firsts",
    "tie_count", "shared_bills", "net_temporal", "balanced",
]
RANKED_COLS = ["company", "rank", "bill_number", "total_amount", "budget_fraction"]


def _load():
    """Load edge CSV, ranked CSV, and GML; raise clearly if any are missing."""
    edges  = pd.read_csv(EDGES_CSV)
    ranked = pd.read_csv(RANKED_CSV)
    G      = nx.read_gml(str(GML_PATH))
    return edges, ranked, G


def check_edge_schema(edges):
    """Check 1: edge CSV has all required columns with expected dtypes."""
    missing = [c for c in EDGE_COLS if c not in edges.columns]
    if missing:
        return False, f"missing columns: {missing}"
    if not pd.api.types.is_float_dtype(edges["weight"]):
        return False, "weight column is not float"
    for col in ("source_firsts", "target_firsts", "tie_count",
                "shared_bills", "net_temporal", "balanced"):
        if not pd.api.types.is_integer_dtype(edges[col]):
            return False, f"{col} is not integer dtype"
    return True, f"{len(edges):,} rows, {len(edges.columns)} columns"


def check_weight_range(edges):
    """Check 2: all RBO weights in (0, 1]."""
    bad = edges[(edges["weight"] <= 0) | (edges["weight"] > 1)]
    if not bad.empty:
        return False, f"{len(bad)} weights outside (0,1]: {bad['weight'].describe()}"
    return True, f"min={edges['weight'].min():.6f}  max={edges['weight'].max():.6f}"


def check_direction_integrity(edges):
    """Check 3: directed edges (balanced=0) always have source_firsts > target_firsts."""
    directed = edges[edges["balanced"] == 0]
    bad      = directed[directed["source_firsts"] <= directed["target_firsts"]]
    if not bad.empty:
        return False, f"{len(bad)} directed edges where source_firsts <= target_firsts"
    return True, f"{len(directed):,} directed edges all have source_firsts > target_firsts"


def check_net_temporal(edges):
    """Check 4: net_temporal == source_firsts - target_firsts for every row."""
    expected = edges["source_firsts"] - edges["target_firsts"]
    mismatch = (edges["net_temporal"] != expected).sum()
    if mismatch:
        return False, f"{mismatch} rows where net_temporal != source_firsts - target_firsts"
    return True, "all rows consistent"


def check_balanced_uniqueness(edges):
    """Check 5: each firm pair appears at most once; balanced edges run source < target (alphabetical)."""
    bal = edges[edges["balanced"] == 1].copy()
    if bal.empty:
        return True, "no balanced edges (vacuously true)"

    # Each pair should appear exactly once (no antiparallel duplicate)
    pairs = list(zip(bal["source"], bal["target"]))
    seen, dupes = set(), []
    for p in pairs:
        if p in seen or (p[1], p[0]) in seen:
            dupes.append(p)
        seen.add(p)
    if dupes:
        return False, f"{len(dupes)} duplicate balanced pairs (antiparallel still present): {dupes[:3]}"

    # Canonical direction: source < target alphabetically
    wrong_dir = bal[bal["source"] > bal["target"]]
    if not wrong_dir.empty:
        sample = list(zip(wrong_dir["source"], wrong_dir["target"]))[:3]
        return False, f"{len(wrong_dir)} balanced edges with source > target (non-canonical): {sample}"

    return True, f"{len(bal):,} balanced edges; each pair once, all source < target"


def check_balanced_zero_net(edges):
    """Check 6: all balanced edges have net_temporal == 0."""
    bal = edges[edges["balanced"] == 1]
    if bal.empty:
        return True, "no balanced edges (vacuously true)"
    bad = bal[bal["net_temporal"] != 0]
    if not bad.empty:
        return False, f"{len(bad)} balanced edges with non-zero net_temporal"
    return True, "all balanced edges have net_temporal == 0"


def check_node_net_influence(G):
    """Check 7: node net_influence equals (out_sf + in_tf) - (out_tf + in_sf) from edge data."""
    errors = []
    for node in G.nodes():
        out_sf = sum(d["source_firsts"] for _, _, d in G.out_edges(node, data=True))
        out_tf = sum(d["target_firsts"] for _, _, d in G.out_edges(node, data=True))
        in_sf  = sum(d["source_firsts"] for _, _, d in G.in_edges(node, data=True))
        in_tf  = sum(d["target_firsts"] for _, _, d in G.in_edges(node, data=True))
        expected = (out_sf + in_tf) - (out_tf + in_sf)
        stored   = G.nodes[node].get("net_influence")
        if stored != expected:
            errors.append((node, stored, expected))
    if errors:
        return False, f"{len(errors)} nodes with wrong net_influence: {errors[:3]}"
    return True, f"all {G.number_of_nodes()} nodes consistent"


def check_node_colors(G):
    """Check 8: node color matches sign of net_influence (green/red/gray)."""
    errors = []
    for node in G.nodes():
        net   = G.nodes[node].get("net_influence", 0)
        color = G.nodes[node].get("color", "")
        expected = "green" if net > 0 else ("red" if net < 0 else "gray")
        if color != expected:
            errors.append((node, color, expected, net))
    if errors:
        return False, f"{len(errors)} nodes with wrong color: {errors[:3]}"
    return True, f"all {G.number_of_nodes()} nodes correctly colored"


def check_ranked_csv(ranked):
    """Check 9: ranked CSV schema OK; each firm has contiguous ranks starting at 1."""
    missing = [c for c in RANKED_COLS if c not in ranked.columns]
    if missing:
        return False, f"missing columns: {missing}"
    errors = []
    for firm, grp in ranked.groupby("company"):
        ranks = sorted(grp["rank"].tolist())
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            errors.append((firm, ranks[:5]))
    if errors:
        return False, f"{len(errors)} firms with non-contiguous ranks: {errors[:2]}"
    n_firms = ranked["company"].nunique()
    return True, (f"{n_firms} firms, {len(ranked):,} rows; "
                  f"max rank per firm: {ranked.groupby('company')['rank'].max().max()}")


def check_net_strength(G):
    """
    Check 11: node net_strength == out_strength(directed) - in_strength(directed).

    Balanced edges (balanced=1) are intentionally excluded from net_strength because
    their canonical direction is alphabetical (arbitrary) and would produce a spurious
    RBO signal unrelated to actual temporal influence.
    """
    errors = []
    for node in G.nodes():
        dir_out = sum(
            (d["weight"] for _, _, d in G.out_edges(node, data=True) if d.get("balanced") == 0),
            0.0,
        )
        dir_in = sum(
            (d["weight"] for _, _, d in G.in_edges(node, data=True) if d.get("balanced") == 0),
            0.0,
        )
        expected = round(dir_out - dir_in, 4)
        stored   = G.nodes[node].get("net_strength")
        if stored is None:
            errors.append((node, "missing", expected))
        elif abs(stored - expected) > 1e-3:
            errors.append((node, stored, expected))
    if errors:
        return False, f"{len(errors)} nodes with wrong net_strength: {errors[:3]}"
    return True, f"all {G.number_of_nodes()} nodes consistent"


def check_rbo_spot(edges, ranked_csv, n_samples=20, seed=42):
    """Check 10: recompute RBO for n_samples random directed pairs; values match CSV."""
    # Reconstruct ranked dict from CSV
    ranked_dict = (
        ranked_csv.sort_values(["company", "rank"])
        .groupby("company")["bill_number"]
        .apply(list)
        .to_dict()
    )
    # Sample from directed edges (balanced=0) to test a mix of weights
    directed = edges[edges["balanced"] == 0].copy()
    if len(directed) == 0:
        return False, "no directed edges to sample"
    sample = directed.sample(min(n_samples, len(directed)), random_state=seed)
    errors = []
    for _, row in sample.iterrows():
        src, tgt = row["source"], row["target"]
        if src not in ranked_dict or tgt not in ranked_dict:
            continue   # firm not in ranked CSV (should not happen)
        recomputed = rbo_score(ranked_dict[src], ranked_dict[tgt], p=0.85)
        if abs(recomputed - row["weight"]) > 1e-4:
            errors.append((src, tgt, row["weight"], recomputed))
    if errors:
        return False, f"{len(errors)} RBO mismatches: {errors[:3]}"
    return True, f"{len(sample)} random pairs recomputed; all within 1e-4"


def main():
    print("=" * 62)
    print("  10 — Congress Influence Network Validation")
    print("=" * 62)

    try:
        edges, ranked, G = _load()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Run rbo_directed_influence.py first.")
        sys.exit(1)

    checks = [
        ("Edge CSV schema",             lambda: check_edge_schema(edges)),
        ("RBO weight range",            lambda: check_weight_range(edges)),
        ("Direction integrity",         lambda: check_direction_integrity(edges)),
        ("net_temporal arithmetic",     lambda: check_net_temporal(edges)),
        ("Balanced edge uniqueness",    lambda: check_balanced_uniqueness(edges)),
        ("Balanced net_temporal=0",     lambda: check_balanced_zero_net(edges)),
        ("Node net_influence math",     lambda: check_node_net_influence(G)),
        ("Node color consistency",      lambda: check_node_colors(G)),
        ("Ranked-list CSV integrity",   lambda: check_ranked_csv(ranked)),
        ("RBO spot-check (n=20)",       lambda: check_rbo_spot(edges, ranked)),
        ("Node net_strength math",      lambda: check_net_strength(G)),
    ]

    passed = 0
    for i, (name, fn) in enumerate(checks, start=1):
        ok, detail = fn()
        status = PASS if ok else FAIL
        print(f"\n  Check {i:2d}: {name}")
        print(f"  {status}  —  {detail}")
        if ok:
            passed += 1

    print(f"\n{'='*62}")
    print(f"  {passed} / {len(checks)} checks passed")
    if passed < len(checks):
        print("  *** Some checks FAILED — review output above ***")
    print("=" * 62)


if __name__ == "__main__":
    main()
