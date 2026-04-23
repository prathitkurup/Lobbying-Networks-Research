"""
Validation 20: net_strength stability and strategic complementarity persistence
across consecutive congressional sessions (111th–117th).

For each adjacent congress pair (c_N → c_{N+1}):

  1. Global top-10 stability
     — Jaccard of top-10 firms by net_strength between sessions.
     — Spearman ρ of net_strength ranks on the common firm set.

  2. Community top-5 stability
     — Per community: Jaccard of top-5 firms by net_strength between sessions.
     — Uses fixed 116th-Congress Leiden partition throughout.

  3. Strategic complementarity persistence
     — For decisive pairs (A leads B, net_temporal > 0) in c_N, grouped by RBO quartile:
         persistence rate (A still leads B in c_{N+1}) and mean RBO in c_{N+1}.
     — Fisher's exact test: Q4 (high-RBO) vs Q1 (low-RBO) × persist vs other.
     — Spearman ρ of RBO values across consecutive sessions.
     — Hypothesis: high-similarity followers (high RBO in c_N) should persist more,
       consistent with strategic complementarity (BCZ 2006).

Requires multi_congress_pipeline.py to have been run for all 7 congresses.
Community labels: fixed 116th-Congress Leiden partition (communities_affiliation.csv).

Run from src/:
  python validations/20_influence_stability_complementarity.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR  = ROOT / "outputs" / "validation"
TXT_PATH = OUT_DIR / "20_influence_stability_complementarity.txt"
CSV_PATH = OUT_DIR / "20_stability_summary.csv"

CONGRESSES = [111, 112, 113, 114, 115, 116, 117]

COMMUNITY_LABELS = {
    0: "Finance/Insurance",
    1: "Tech/Telecom",
    2: "Defense/Industrial",
    3: "Energy/Utilities",
    4: "Health/Pharma",
}

TOP_K_GLOBAL = 10   # top-K global firms for net_strength stability
TOP_K_COMM   = 5    # top-K per community for net_strength stability


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

def load_congress_data():
    """Load node_attributes.csv and rbo_directed_influence.csv for all congresses."""
    nodes, edges = {}, {}
    for c in CONGRESSES:
        npath = DATA_DIR / f"congress/{c}/node_attributes.csv"
        epath = DATA_DIR / f"congress/{c}/rbo_directed_influence.csv"
        if not npath.exists() or not epath.exists():
            raise FileNotFoundError(
                f"Missing data for {c}th Congress. Run multi_congress_pipeline.py first."
            )
        nodes[c] = pd.read_csv(npath)
        edges[c] = pd.read_csv(epath)
    return nodes, edges


def load_community_partition():
    """Return {firm: community_id} from fixed 116th-Congress Leiden partition."""
    df = pd.read_csv(DATA_DIR / "archive" / "communities" / "communities_affiliation.csv")
    return dict(zip(df["fortune_name"], df["community_aff"]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jaccard(s1, s2):
    """Jaccard similarity of two sets."""
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2:
        return np.nan
    return round(len(s1 & s2) / len(s1 | s2), 4)


def top_k_firms(nodes_df, k, metric="net_strength"):
    """Return set of top-k firm names by metric."""
    return set(nodes_df.nlargest(k, metric)["firm"].tolist())


def spearman_rank_stability(nodes_n, nodes_n1, metric="net_strength"):
    """Spearman rho of metric ranks between sessions, on the common firm set."""
    merged = nodes_n[["firm", metric]].merge(
        nodes_n1[["firm", metric]], on="firm", suffixes=("_n", "_n1")
    )
    if len(merged) < 5:
        return np.nan, np.nan, len(merged)
    rho, pval = stats.spearmanr(merged[f"{metric}_n"], merged[f"{metric}_n1"])
    return round(float(rho), 4), round(float(pval), 6), len(merged)


# ---------------------------------------------------------------------------
# Analysis 1: global top-K stability
# ---------------------------------------------------------------------------

def global_top_k_stability(nodes_n, nodes_n1, c_n, c_n1, k=TOP_K_GLOBAL):
    """
    Jaccard and Spearman rho of top-k net_strength firms across two sessions.
    Returns a result dict with the top-k DataFrames for printing.
    """
    top_n  = top_k_firms(nodes_n,  k)
    top_n1 = top_k_firms(nodes_n1, k)
    j = jaccard(top_n, top_n1)

    top_n_df  = nodes_n.nlargest(k, "net_strength")[["firm", "net_strength", "net_influence"]].copy()
    top_n1_df = nodes_n1.nlargest(k, "net_strength")[["firm", "net_strength", "net_influence"]].copy()

    rho, pval, n_common = spearman_rank_stability(nodes_n, nodes_n1)

    return {
        "jaccard":        j,
        "spearman_rho":   rho,
        "spearman_pval":  pval,
        "n_common":       n_common,
        "top_n":          top_n,
        "top_n1":         top_n1,
        "top_n_df":       top_n_df,
    }


# ---------------------------------------------------------------------------
# Analysis 2: community top-K stability
# ---------------------------------------------------------------------------

def community_top_k_stability(nodes_n, nodes_n1, partition, k=TOP_K_COMM):
    """Per-community top-k Jaccard across two sessions, using fixed 116th partition."""
    results = {}
    for cid, clabel in COMMUNITY_LABELS.items():
        members = {f for f, c in partition.items() if c == cid}
        sub_n   = nodes_n[nodes_n["firm"].isin(members)]
        sub_n1  = nodes_n1[nodes_n1["firm"].isin(members)]

        if len(sub_n) < k or len(sub_n1) < k:
            results[cid] = {"label": clabel, "jaccard": np.nan, "top_n": set(), "top_n1": set()}
            continue

        top_n  = top_k_firms(sub_n,  k)
        top_n1 = top_k_firms(sub_n1, k)
        results[cid] = {
            "label":   clabel,
            "jaccard": jaccard(top_n, top_n1),
            "top_n":   top_n,
            "top_n1":  top_n1,
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 3: strategic complementarity persistence
# ---------------------------------------------------------------------------

def complementarity_test(edges_n, edges_n1):
    """
    For decisive pairs in c_N (A leads B; canonical source < target, net_temporal > 0):
      - Look up whether A still leads B in c_{N+1} (direction persistence).
      - Group by RBO quartile in c_N; compute persistence rate and mean RBO in c_{N+1}.
      - Fisher's exact test: Q4 (high-RBO) vs Q1 (low-RBO) × persist vs other.
      - Spearman rho of RBO values across sessions for pairs present in both.

    Returns (merged_df, quartile_agg, fisher_result_dict_or_None, rbo_rho, rbo_pval).
    Returns None for merged_df if edge CSV lacks rbo column (pre-pipeline-rerun).
    """
    if "rbo" not in edges_n.columns or "rbo" not in edges_n1.columns:
        print("  WARNING: edge CSV lacks 'rbo' column — "
              "complementarity test skipped. Re-run multi_congress_pipeline.py.")
        return None, None, None, np.nan, np.nan

    # Canonical decisive pairs in c_N: source < target, source is net first-mover
    decisive_n = edges_n[
        (edges_n["source"] < edges_n["target"]) & (edges_n["net_temporal"] > 0)
    ][["source", "target", "rbo", "net_temporal"]].copy()
    decisive_n.columns = ["source", "target", "rbo_n", "nt_n"]

    # Lookup in c_{N+1}: same canonical form
    canon_n1 = edges_n1[edges_n1["source"] < edges_n1["target"]].copy()
    lookup = (
        canon_n1.set_index(["source", "target"])[["rbo", "net_temporal"]]
        .rename(columns={"rbo": "rbo_n1", "net_temporal": "nt_n1"})
    )
    merged = decisive_n.join(lookup, on=["source", "target"], how="left")

    merged["in_n1"] = merged["nt_n1"].notna()
    merged["persists"] = merged["nt_n1"] > 0
    merged["direction_n1"] = np.where(
        merged["nt_n1"] > 0, "persist",
        np.where(merged["nt_n1"] < 0, "reverse",
                 np.where(merged["nt_n1"] == 0, "tied", "absent"))
    )

    # RBO quartile in c_N
    merged["rbo_quartile"] = pd.qcut(
        merged["rbo_n"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )

    q_agg = (
        merged.groupby("rbo_quartile", observed=True)
        .agg(
            n_pairs        = ("source", "count"),
            n_in_n1        = ("in_n1",  "sum"),
            n_persist      = ("persists", "sum"),
            mean_rbo_n     = ("rbo_n",   "mean"),
            mean_rbo_n1    = ("rbo_n1",  "mean"),
        )
        .assign(
            persist_rate = lambda x: x["n_persist"] / x["n_in_n1"].clip(lower=1),
            pct_in_n1    = lambda x: x["n_in_n1"]  / x["n_pairs"],
        )
    )

    # Fisher's exact: Q4 vs Q1 × persist vs other
    high = merged[merged["rbo_quartile"] == "Q4 (high)"].dropna(subset=["nt_n1"])
    low  = merged[merged["rbo_quartile"] == "Q1 (low)"].dropna(subset=["nt_n1"])

    fisher_result = None
    if len(high) >= 3 and len(low) >= 3:
        contingency = np.array([
            [int(high["persists"].sum()),  len(high) - int(high["persists"].sum())],
            [int(low["persists"].sum()),   len(low)  - int(low["persists"].sum())],
        ])
        _, pval_fisher = stats.fisher_exact(contingency, alternative="greater")
        fisher_result = {
            "high_n":            len(high),
            "high_persist":      int(high["persists"].sum()),
            "high_persist_rate": float(high["persists"].mean()),
            "low_n":             len(low),
            "low_persist":       int(low["persists"].sum()),
            "low_persist_rate":  float(low["persists"].mean()),
            "fisher_pval":       round(float(pval_fisher), 5),
        }

    # Spearman rho of RBO across sessions (decisive pairs present in both)
    both = merged.dropna(subset=["rbo_n1"])
    rbo_rho = rbo_pval = np.nan
    if len(both) >= 5:
        rbo_rho, rbo_pval = stats.spearmanr(both["rbo_n"], both["rbo_n1"])
        rbo_rho  = round(float(rbo_rho),  4)
        rbo_pval = round(float(rbo_pval), 6)

    return merged, q_agg, fisher_result, rbo_rho, rbo_pval


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _orig = sys.stdout
    _f    = open(TXT_PATH, "w")
    sys.stdout = _Tee(_orig, _f)

    try:
        SEP = "=" * 72

        print(SEP)
        print("VALIDATION 20: INFLUENCE STABILITY & STRATEGIC COMPLEMENTARITY")
        print("             111th–117th Congress — consecutive session pairs")
        print(SEP)
        print("\nPrimary metric : net_strength = Σ_j [RBO(i,j) × net_temporal(i,j)]")
        print("Reference metric: net_influence (unweighted first-mover count)")
        print("Communities    : fixed 116th-Congress Leiden partition")
        print(f"\nGlobal top-K: {TOP_K_GLOBAL}   Community top-K: {TOP_K_COMM}")

        # -- Load data -------------------------------------------------------
        print("\n[1/4] Loading congress data ...")
        try:
            nodes, edges = load_congress_data()
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            sys.exit(1)

        partition = load_community_partition()

        for c in CONGRESSES:
            n_firms = len(nodes[c])
            n_edges = len(edges[c])
            print(f"  {c}th: {n_firms:>4} firms,  {n_edges:>6,} edges")

        # -- Consecutive pairs loop ------------------------------------------
        print("\n[2/4] Analyzing consecutive session pairs ...")

        summary_rows = []

        for i in range(len(CONGRESSES) - 1):
            c_n, c_n1 = CONGRESSES[i], CONGRESSES[i + 1]

            print(f"\n{SEP}")
            print(f"  {c_n}th Congress  →  {c_n1}th Congress")
            print(SEP)

            # ── 1. Global top-K stability ─────────────────────────────────
            print(f"\n  1. GLOBAL TOP-{TOP_K_GLOBAL} NET_STRENGTH STABILITY")
            print("  " + "-" * 58)

            g = global_top_k_stability(nodes[c_n], nodes[c_n1], c_n, c_n1)

            in_both = g["top_n"] & g["top_n1"]
            only_n  = g["top_n"] - g["top_n1"]
            only_n1 = g["top_n1"] - g["top_n"]

            print(f"\n  Top-{TOP_K_GLOBAL} in {c_n}th  ('+' = retained in {c_n1}th):")
            print(f"  {'':2} {'Firm':<43} {'net_str':>8}  {'net_inf':>8}")
            for _, row in g["top_n_df"].iterrows():
                mark = "+" if row["firm"] in g["top_n1"] else " "
                print(f"  {mark} {row['firm']:<43} {row['net_strength']:>8.4f}  "
                      f"{int(row['net_influence']):>8}")

            print(f"\n  Jaccard (top-{TOP_K_GLOBAL}):         {g['jaccard']:.4f}  "
                  f"({len(in_both)}/{TOP_K_GLOBAL} firms retained)")
            print(f"  Spearman rho (net_strength): rho={g['spearman_rho']:.4f}  "
                  f"p={g['spearman_pval']:.4e}  n={g['n_common']} common firms")
            if only_n:
                print(f"  Leaving  top-{TOP_K_GLOBAL}: {', '.join(sorted(only_n))}")
            if only_n1:
                print(f"  Entering top-{TOP_K_GLOBAL}: {', '.join(sorted(only_n1))}")

            # ── 2. Community top-K stability ──────────────────────────────
            print(f"\n  2. COMMUNITY TOP-{TOP_K_COMM} NET_STRENGTH STABILITY")
            print("  " + "-" * 58)

            comm_results = community_top_k_stability(nodes[c_n], nodes[c_n1], partition)

            print(f"\n  {'Community':<22} {'Jaccard':>8}  {'Retained':>9}  New entrants in {c_n1}th")
            print(f"  {'-'*72}")
            mean_comm_j = []
            for cid, cr in comm_results.items():
                if np.isnan(cr["jaccard"]):
                    print(f"  {cr['label']:<22} {'—':>8}  (insufficient members)")
                    continue
                in_both_c = cr["top_n"] & cr["top_n1"]
                only_n1_c = cr["top_n1"] - cr["top_n"]
                entering  = ", ".join(sorted(only_n1_c)[:3]) if only_n1_c else "none"
                if len(only_n1_c) > 3:
                    entering += f" +{len(only_n1_c)-3} more"
                print(f"  {cr['label']:<22} {cr['jaccard']:>8.4f}  "
                      f"{len(in_both_c)}/{TOP_K_COMM}  {entering}")
                mean_comm_j.append(cr["jaccard"])

            mean_cj = round(np.mean(mean_comm_j), 4) if mean_comm_j else np.nan
            print(f"\n  Mean community Jaccard: {mean_cj:.4f}")

            # ── 3. Strategic complementarity persistence ──────────────────
            print(f"\n  3. STRATEGIC COMPLEMENTARITY PERSISTENCE")
            print("  " + "-" * 58)

            merged, q_agg, fisher, rbo_rho, rbo_pval = complementarity_test(
                edges[c_n], edges[c_n1]
            )

            if merged is None:
                # rbo column absent — complementarity test skipped for this pair
                pass
            else:
                n_decisive = len(merged)
                n_in_both  = int(merged["in_n1"].sum())
                n_persist  = int(merged["persists"].sum())

                print(f"\n  Decisive pairs in {c_n}th:  {n_decisive:,}")
                print(f"  Present in {c_n1}th:        {n_in_both:,}  "
                      f"({100*n_in_both/max(n_decisive, 1):.1f}%)")
                print(f"  Direction persists:      {n_persist:,}  "
                      f"({100*n_persist/max(n_in_both, 1):.1f}% of pairs present in {c_n1}th)")

                print(f"\n  Outcome breakdown (among {c_n}th decisive pairs):")
                for outcome in ["persist", "reverse", "tied", "absent"]:
                    cnt = (merged["direction_n1"] == outcome).sum()
                    pct = 100 * cnt / max(n_decisive, 1)
                    print(f"    {outcome:<10}: {cnt:>5}  ({pct:.1f}%)")

                print(f"\n  RBO persistence (Spearman rho across sessions): "
                      f"rho={rbo_rho:.4f}  p={rbo_pval:.4e}")

                print(f"\n  Direction persistence by RBO quartile in {c_n}th:")
                print(f"  {'Quartile':<14} {'n_pairs':>8} {'in_n1':>7} "
                      f"{'persist':>9} {'persist_%':>10} {'mean_rbo_n':>11} {'mean_rbo_n1':>12}")
                print(f"  {'-'*73}")
                for q_label, qrow in q_agg.iterrows():
                    n1_str = f"{qrow['mean_rbo_n1']:.4f}" if pd.notna(qrow["mean_rbo_n1"]) else "—"
                    print(f"  {str(q_label):<14} {int(qrow['n_pairs']):>8} "
                          f"{int(qrow['n_in_n1']):>7} {int(qrow['n_persist']):>9} "
                          f"{100*qrow['persist_rate']:>9.1f}% "
                          f"{qrow['mean_rbo_n']:>11.4f} {n1_str:>12}")

                if fisher:
                    print(f"\n  Fisher's exact (Q4-high vs Q1-low RBO x persist vs other):")
                    print(f"    High-RBO Q4: {fisher['high_persist']}/{fisher['high_n']}  "
                          f"persist  ({100*fisher['high_persist_rate']:.1f}%)")
                    print(f"    Low-RBO  Q1: {fisher['low_persist']}/{fisher['low_n']}  "
                          f"persist  ({100*fisher['low_persist_rate']:.1f}%)")
                    interp = ("consistent with complementarity" if fisher["fisher_pval"] < 0.05
                              else "not significant at p < 0.05")
                    print(f"    One-sided p (H1: high-RBO persists more): "
                          f"{fisher['fisher_pval']:.4f}  — {interp}")

            # Complementarity columns — NaN when edge CSV lacks rbo
            _n_dec    = n_decisive if merged is not None else np.nan
            _n_in     = n_in_both  if merged is not None else np.nan
            _n_per    = n_persist  if merged is not None else np.nan
            _per_rate = round(_n_per / max(_n_in, 1), 4) if merged is not None else np.nan

            summary_rows.append({
                "congress_pair":          f"{c_n}->{c_n1}",
                "global_jaccard":         g["jaccard"],
                "global_spearman_rho":    g["spearman_rho"],
                "global_spearman_pval":   g["spearman_pval"],
                "n_common_firms":         g["n_common"],
                "mean_comm_jaccard":      mean_cj,
                "n_decisive_pairs":       _n_dec,
                "n_in_both":              _n_in,
                "n_persist":              _n_per,
                "overall_persist_rate":   _per_rate,
                "rbo_spearman_rho":       rbo_rho,
                "rbo_spearman_pval":      rbo_pval,
                "fisher_pval_q4_vs_q1":   fisher["fisher_pval"] if fisher else np.nan,
                "high_rbo_persist_rate":  fisher["high_persist_rate"] if fisher else np.nan,
                "low_rbo_persist_rate":   fisher["low_persist_rate"]  if fisher else np.nan,
            })

        # -- Summary table ---------------------------------------------------
        print(f"\n{SEP}")
        print("[3/4] SUMMARY ACROSS ALL CONSECUTIVE SESSION PAIRS")
        print(SEP)

        summary_df = pd.DataFrame(summary_rows)

        print(f"\n  {'Pair':<12} {'GlobJ':>6} {'rho(ns)':>9} {'CommJ':>7} "
              f"{'PersistR':>9} {'RBO-rho':>8} {'Fisher-p':>10}")
        print(f"  {'-'*65}")
        for _, r in summary_df.iterrows():
            fp     = f"{r['fisher_pval_q4_vs_q1']:.4f}" if pd.notna(r["fisher_pval_q4_vs_q1"]) else "—"
            rho_ns = f"{r['global_spearman_rho']:.4f}"  if pd.notna(r["global_spearman_rho"])  else "—"
            cj     = f"{r['mean_comm_jaccard']:.4f}"    if pd.notna(r["mean_comm_jaccard"])     else "—"
            pr     = f"{r['overall_persist_rate']:.3f}" if pd.notna(r["overall_persist_rate"])  else "—"
            rr     = f"{r['rbo_spearman_rho']:.4f}"     if pd.notna(r["rbo_spearman_rho"])      else "—"
            print(f"  {r['congress_pair']:<12} {r['global_jaccard']:>6.4f} "
                  f"{rho_ns:>9} {cj:>7} {pr:>9} {rr:>8} {fp:>10}")

        print(f"\n  Mean global Jaccard:       {summary_df['global_jaccard'].mean():.4f}")
        print(f"  Mean Spearman rho(ns):     {summary_df['global_spearman_rho'].mean():.4f}")
        print(f"  Mean community Jaccard:    {summary_df['mean_comm_jaccard'].mean():.4f}")
        print(f"  Mean persist rate:         {summary_df['overall_persist_rate'].mean():.3f}")
        print(f"  Mean RBO Spearman rho:     {summary_df['rbo_spearman_rho'].mean():.4f}")

        n_sig = (summary_df["fisher_pval_q4_vs_q1"] < 0.05).sum()
        print(f"  Fisher p < 0.05 (complementarity supported): "
              f"{n_sig}/{len(summary_df)} session pairs")

        print(f"\n  Interpretation (complementarity hypothesis):")
        mean_pr = summary_df["overall_persist_rate"].mean()
        mean_hr = summary_df["high_rbo_persist_rate"].mean()
        mean_lr = summary_df["low_rbo_persist_rate"].mean()
        print(f"  Overall direction persistence rate: {100*mean_pr:.1f}%")
        if not (np.isnan(mean_hr) or np.isnan(mean_lr)):
            diff = mean_hr - mean_lr
            print(f"  High-RBO (Q4) mean persist rate:    {100*mean_hr:.1f}%")
            print(f"  Low-RBO  (Q1) mean persist rate:    {100*mean_lr:.1f}%")
            print(f"  Differential:                      {100*diff:+.1f}pp — "
                  + ("consistent with complementarity" if diff > 0.05
                     else "weak or no differential"))

        # -- Save outputs ----------------------------------------------------
        print(f"\n[4/4] Saving outputs ...")
        summary_df.to_csv(CSV_PATH, index=False)
        print(f"  Summary CSV  -> {CSV_PATH.name}")
        print(f"  Log          -> {TXT_PATH.name}")

        print(f"\n{SEP}")
        print("Validation complete.")
        print(SEP)

    finally:
        sys.stdout = _orig
        _f.close()


if __name__ == "__main__":
    main()
