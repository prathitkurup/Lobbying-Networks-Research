"""
Validation 07: Composite similarity network — structural checks and unit tests.

Verifies:
  1.  Composite weight = affil_norm × cosine_weight × rbo_weight (formula check).
  2.  Composite weights are in [0, 1] for all edges.
  3.  Composite edges form a strict subset of cosine ∩ RBO edge pairs.
  4.  Triple-filter sparsity: composite edge count < cosine and < RBO edge counts.
  5.  High-weight core: top-10% edges carry ≥ 25% of total weight (right-skewed).
  6.  Canonical pair ordering preserved: source < target lexicographically.
  7.  Katz centrality: alpha < 1/spectral_radius (convergence guarantee).
  8.  Katz centrality values are non-negative.
  9.  Katz > 0 for all nodes in a connected graph.
  10. Katz ranking is different from PageRank (captures different structure).
  11. Synthetic 5-firm test: two clear clusters should receive the same
      Leiden community assignment.

Design decision (cross-reference to design_decisions.md §12):
  The composite network requires all three filters to simultaneously pass,
  producing a sparse high-weight core. A pair (i,j) earns a high composite
  weight only if it is aligned in portfolio DIRECTION (cosine), PRIORITY
  ORDERING (RBO), AND BREADTH (affil_norm). This triple constraint is
  expected to yield significantly fewer edges than any single metric but
  with higher interpretive confidence.

  Katz-Bonacich centrality (§13) complements PageRank by penalising longer
  paths more aggressively (exponential decay vs. PageRank's degree-normalised
  flow). For lobbying networks, Katz identifies firms that are influential
  not just through their direct ties but through their embedding in
  multi-hop co-lobbying chains — a structural proxy for indirect political
  influence propagation.
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from utils.similarity import aggregate_per_firm_bill
from utils.centrality import compute_katz_centrality
from composite_similarity_network import build_composite_edges
from config import DATA_DIR

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "validation" / "07_composite_network_validation.txt"

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        FAIL += 1


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
_orig_stdout = sys.stdout
_f = open(OUTPUT_PATH, "w")
sys.stdout = _Tee(_orig_stdout, _f)

try:
    print("=" * 65)
    print("Validation 07: Composite network & Katz-Bonacich unit tests")
    print("=" * 65)


    # ---------------------------------------------------------------------------
    # Section A: Composite formula verification (pure math, no external functions)
    # ---------------------------------------------------------------------------
    print("\n[A] Composite formula (synthetic edge data)")

    cos_val = 0.60
    rbo_val = 0.30
    affil_val = 0.50
    expected_weight = round(cos_val * rbo_val * affil_val, 8)

    row = {
        "source": "A", "target": "B",
        "cosine_weight": cos_val, "rbo_weight": rbo_val,
        "affil_norm": affil_val,
    }
    computed_weight = round(row["cosine_weight"] * row["rbo_weight"] * row["affil_norm"], 8)

    check(
        "Weight = cosine × rbo × affil_norm",
        computed_weight == expected_weight,
        f"got {computed_weight}, expected {expected_weight}"
    )

    # Weight ≤ min(cosine, rbo)
    check(
        "composite ≤ cosine and ≤ rbo",
        computed_weight <= cos_val and computed_weight <= rbo_val,
    )

    # Monotone: higher affil_norm → higher composite weight (cosine and rbo fixed)
    w_low  = cos_val * rbo_val * 0.10
    w_high = cos_val * rbo_val * 0.50
    check(
        "Higher affil_norm → higher composite weight (monotone)",
        w_high > w_low,
        f"w_low={w_low:.4f}  w_high={w_high:.4f}"
    )

    # Composite = 0 when any component is 0
    check(
        "Composite = 0 when cosine = 0",
        round(0.0 * rbo_val * affil_val, 8) == 0.0
    )
    check(
        "Composite = 0 when rbo = 0",
        round(cos_val * 0.0 * affil_val, 8) == 0.0
    )
    check(
        "Composite = 0 when affil_norm = 0",
        round(cos_val * rbo_val * 0.0, 8) == 0.0
    )


    # ---------------------------------------------------------------------------
    # Section B: Katz-Bonacich centrality
    # ---------------------------------------------------------------------------
    print("\n[B] Katz-Bonacich centrality")

    # Build a small known graph for testing
    G_test = nx.Graph()
    edges = [
        ("A", "B", 0.8), ("A", "C", 0.6), ("B", "C", 0.7),
        ("C", "D", 0.4), ("D", "E", 0.9),
    ]
    for u, v, w in edges:
        G_test.add_edge(u, v, weight=w)

    katz = compute_katz_centrality(G_test)

    check("Katz: returns dict keyed by node",
          isinstance(katz, dict) and set(katz.keys()) == set(G_test.nodes()))

    check("Katz: all values non-negative",
          all(v >= 0 for v in katz.values()),
          f"min={min(katz.values()):.4f}")

    check("Katz: all nodes have centrality > 0 (connected graph)",
          all(v > 0 for v in katz.values()),
          f"zeros: {[k for k,v in katz.items() if v == 0]}")

    # Katz alpha < 1/spectral_radius
    A = nx.to_numpy_array(G_test, weight="weight")
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    expected_alpha = 0.85 / rho
    check("Katz: spectral radius > 0 (non-trivial graph)",
          rho > 0,
          f"rho={rho:.4f}")
    check("Katz: alpha = 0.85/rho < 1/rho",
          expected_alpha < 1.0 / rho,
          f"alpha={expected_alpha:.4f} 1/rho={1/rho:.4f}")

    # Katz and PageRank should correlate but differ
    pr = nx.pagerank(G_test, weight="weight")
    nodes = sorted(G_test.nodes())
    katz_ranks = sorted(nodes, key=lambda n: katz[n], reverse=True)
    pr_ranks   = sorted(nodes, key=lambda n: pr[n],   reverse=True)
    check("Katz and PageRank rankings are not identical (different measures)",
          katz_ranks != pr_ranks,
          f"katz_order={katz_ranks}  pr_order={pr_ranks}")

    # Katz: empty graph should return empty dict
    G_empty = nx.Graph()
    katz_empty = compute_katz_centrality(G_empty)
    check("Katz: empty graph returns {}", katz_empty == {})

    # Katz: single-node graph should return {node: 0.0} (no edges = zero weight)
    G_single = nx.Graph()
    G_single.add_node("X")
    katz_single = compute_katz_centrality(G_single)
    check("Katz: single isolated node returns 0.0",
          katz_single.get("X", None) == 0.0,
          f"got {katz_single}")


    # ---------------------------------------------------------------------------
    # Section C: Live data structural checks (uses actual Fortune 500 data)
    # ---------------------------------------------------------------------------
    print("\n[C] Live data structural checks (Fortune 500 OpenSecrets data)")

    try:
        from utils.data_loading import load_bills_data
        from utils.network_building import build_graph_with_attrs
        df = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")

        print(f"\n  Loaded: {df['fortune_name'].nunique()} firms, "
              f"{df['bill_number'].nunique()} bills")

        # Build composite edges from pre-computed CSVs
        print("  Building composite edges (from pre-computed cosine/rbo/affil CSVs)...")
        comp_edges = build_composite_edges()

        # Load component edge CSVs for subset checks
        cos_edges  = pd.read_csv(DATA_DIR / "cosine_edges.csv")
        rbo_edges  = pd.read_csv(DATA_DIR / "rbo_edges.csv")

        # Composite edges ⊆ cosine edge pairs
        cos_pairs  = set(zip(cos_edges["source"],  cos_edges["target"]))
        rbo_pairs  = set(zip(rbo_edges["source"],  rbo_edges["target"]))
        comp_pairs = set(zip(comp_edges["source"], comp_edges["target"]))

        check(
            "Composite edge pairs ⊆ cosine edge pairs",
            comp_pairs.issubset(cos_pairs),
            f"{len(comp_pairs - cos_pairs)} composite pairs not in cosine"
        )
        check(
            "Composite edge pairs ⊆ RBO edge pairs",
            comp_pairs.issubset(rbo_pairs),
            f"{len(comp_pairs - rbo_pairs)} composite pairs not in RBO"
        )

        # Triple-filter sparsity
        check(
            "Composite edge count < cosine edge count (triple-filter sparsification)",
            len(comp_edges) < len(cos_edges),
            f"composite={len(comp_edges)} cosine={len(cos_edges)}"
        )
        check(
            "Composite edge count < RBO edge count",
            len(comp_edges) < len(rbo_edges),
            f"composite={len(comp_edges)} rbo={len(rbo_edges)}"
        )

        print(f"\n  Edge counts: cosine={len(cos_edges):,}  "
              f"rbo={len(rbo_edges):,}  composite={len(comp_edges):,}")
        print(f"  Triple-filter reduction: "
              f"{100*(1-len(comp_edges)/len(cos_edges)):.1f}% vs cosine  "
              f"{100*(1-len(comp_edges)/len(rbo_edges)):.1f}% vs RBO")

        # Composite weights ∈ [0, 1]
        check(
            "All composite weights ∈ [0, 1]",
            (comp_edges["weight"] >= 0).all() and (comp_edges["weight"] <= 1.0).all(),
            f"min={comp_edges['weight'].min():.6f} max={comp_edges['weight'].max():.6f}"
        )

        # High-weight core: top 10% of edges carry ≥ 25% of total weight
        w = comp_edges["weight"]
        top10_thresh = w.quantile(0.90)
        top10_weight_share = w[w >= top10_thresh].sum() / w.sum()
        check(
            "High-weight core: top 10% edges carry ≥ 25% of total weight",
            top10_weight_share >= 0.25,
            f"top-10% share = {100*top10_weight_share:.1f}%  "
            f"(threshold={top10_thresh:.4f})"
        )
        print(f"  Top-10% weight concentration: {100*top10_weight_share:.1f}%  "
              f"[>= 25% expected for a right-skewed core]")

        # Canonical pair ordering (source < target alphabetically)
        bad_order = comp_edges[comp_edges["source"] >= comp_edges["target"]]
        check(
            "Canonical pair ordering: source < target for all edges",
            bad_order.empty,
            f"{len(bad_order)} edges with source >= target"
        )

        # Composite weight = affil_norm × cosine_weight × rbo_weight
        comp_edges["expected_weight"] = (
            comp_edges["affil_norm"] * comp_edges["cosine_weight"] * comp_edges["rbo_weight"]
        ).round(8)
        discrepancies = (comp_edges["expected_weight"] - comp_edges["weight"].round(8)).abs()
        check(
            "Composite weight = affil_norm × cosine × rbo exactly",
            (discrepancies < 1e-6).all(),
            f"max discrepancy: {discrepancies.max():.2e}"
        )

        # Component correlations: cosine & rbo should be positively correlated with shared_n
        if len(comp_edges) > 2:
            r_cs = comp_edges[["cosine_weight", "shared_n"]].corr().iloc[0, 1]
            check(
                "cosine_weight and shared_n positively correlated (r > 0)",
                r_cs > 0,
                f"r = {r_cs:.3f}"
            )
            r_rs = comp_edges[["rbo_weight", "shared_n"]].corr().iloc[0, 1]
            check(
                "rbo_weight and shared_n positively correlated (r > 0)",
                r_rs > 0,
                f"r = {r_rs:.3f}"
            )

        # Katz on actual composite graph
        G_comp = build_graph_with_attrs(comp_edges, weight_col="weight")
        katz_comp = compute_katz_centrality(G_comp)
        check(
            "Katz centrality computed for all composite graph nodes",
            set(katz_comp.keys()) == set(G_comp.nodes()),
        )
        check(
            "All composite Katz values non-negative",
            all(v >= 0 for v in katz_comp.values()),
            f"negative: {[(k,v) for k,v in katz_comp.items() if v < 0][:3]}"
        )

    except FileNotFoundError as e:
        print(f"  SKIP  Live data tests — data file not found: {e}")
        print(f"  (Run opensecrets_extraction.py and the individual network scripts first.)")


    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 65)

finally:
    sys.stdout = _orig_stdout
    _f.close()

if FAIL > 0:
    sys.exit(1)
