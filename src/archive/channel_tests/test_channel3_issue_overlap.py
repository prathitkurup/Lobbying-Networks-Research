"""
Channel 3 Test: Issue-Space Correlated Response as Driver of Directed Influence.

Hypothesis: companies in the same regulatory domain independently track and
adopt the same bills — not because one influences the other but because the
same legislation is objectively relevant to both. The RBO temporal precedence
then captures correlated monitoring speed, not transmission.

Tests:
  1. Issue-code similarity distribution: are directed RBO pairs more issue-similar
     than balanced pairs or non-edge pairs?
  2. RBO weight vs issue similarity: does issue alignment predict RBO weight?
  3. Mediation under conditioning: among high-issue-overlap pairs, is affiliation
     mediation even rarer than overall? (If overlap drives correlated response,
     the pairs with most overlap should need no channel at all.)
  4. Issue RBO community vs bill RBO community alignment: do the issue-driven
     communities match the directed influence communities?

Issue similarity is measured two ways:
  cosine  — cosine similarity on spend-fraction vectors across 76 issue codes
  rbo     — RBO similarity on ranked issue-code lists (from issue_rbo_edges.csv)

Outputs:
  outputs/channel_tests/channel3_issue_overlap.txt
  outputs/channel_tests/channel3_issue_overlap.png

Run from src/:  python channel_tests/test_channel3_issue_overlap.py
"""

import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, ROOT, OPENSECRETS_ISSUES_CSV

OUT_DIR = ROOT / "outputs" / "channel_tests"
OUT_TXT = OUT_DIR / "channel3_issue_overlap.txt"
OUT_PNG = OUT_DIR / "channel3_issue_overlap.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.labelsize": 11, "figure.dpi": 150})

NAVY   = "#1F3864"
STEEL  = "#2E5FA3"
AMBER  = "#E67E22"
GREEN  = "#27AE60"
RED    = "#C0392B"
GRAY   = "#7F8C8D"
LGRAY  = "#ECF0F1"


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text): [s.write(text) for s in self.streams]
    def flush(self):       [s.flush()     for s in self.streams]


# -- Data loading and similarity computation ----------------------------------

def build_issue_cosine_matrix():
    """Compute cosine similarity matrix over issue-code spend fractions."""
    df = pd.read_csv(OPENSECRETS_ISSUES_CSV)
    agg = (df.groupby(["fortune_name", "issue_code"])["amount_allocated"]
             .sum().reset_index()
             .rename(columns={"amount_allocated": "total_amount"}))
    firm_totals = agg.groupby("fortune_name")["total_amount"].transform("sum")
    agg = agg[firm_totals > 0].copy()
    agg["frac"] = agg["total_amount"] / firm_totals[firm_totals > 0]

    pivot = (agg.pivot_table(index="fortune_name", columns="issue_code",
                              values="frac", aggfunc="sum")
               .fillna(0))
    firms = pivot.index.tolist()
    codes = pivot.columns.tolist()
    mat   = pivot.values.astype(np.float64)
    cos   = cosine_similarity(mat)
    cos_df = pd.DataFrame(cos, index=firms, columns=firms)
    return cos_df, pivot, firms, codes


def load_rbo_issue_edges():
    """Load issue RBO edges (if available) as an undirected similarity dict."""
    path = DATA_DIR / "network_edges" / "issue_rbo_edges.csv"
    if not path.exists():
        return None
    edges = pd.read_csv(path)
    sim   = {}
    for _, row in edges.iterrows():
        sim[(row["source"], row["target"])] = row["weight"]
        sim[(row["target"], row["source"])] = row["weight"]
    return sim


def build_pair_dataset(rbo_edges, cos_df, issue_rbo_sim, med_df):
    """Construct per-pair dataset joining RBO edges, issue similarity, and mediation."""
    firms = cos_df.index.tolist()
    firm_set = set(firms)

    # Mediation lookup: {(source, target): is_any_mediated, net_any_connected}
    med_lookup = {}
    if med_df is not None:
        med_dir = med_df[med_df["is_bill_directed"]]
        for _, row in med_dir.iterrows():
            key = (row["rbo_source"], row["rbo_target"])
            if key not in med_lookup:
                med_lookup[key] = {"any_mediated": False, "net_connected": False, "n": 0}
            med_lookup[key]["any_mediated"]  |= bool(row["is_any_mediated"])
            med_lookup[key]["net_connected"] |= bool(row["net_any_connected"])
            med_lookup[key]["n"]             += 1

    records = []
    for _, row in rbo_edges.iterrows():
        s, t = row["source"], row["target"]
        if s not in firm_set or t not in firm_set:
            continue
        cos_sim = cos_df.loc[s, t]
        iss_rbo = issue_rbo_sim.get((s, t), 0.0) if issue_rbo_sim else np.nan
        med      = med_lookup.get((s, t), {})
        records.append({
            "source":           s,
            "target":           t,
            "rbo_weight":       row["weight"],
            "balanced":         row["balanced"],
            "net_temporal":     row["net_temporal"],
            "shared_bills":     row["shared_bills"],
            "issue_cos":        cos_sim,
            "issue_rbo":        iss_rbo,
            "any_mediated":     med.get("any_mediated", False),
            "net_connected":    med.get("net_connected", False),
            "n_dir_bills":      med.get("n", 0),
        })
    pairs = pd.DataFrame(records)
    pairs["edge_type"] = pairs["balanced"].map({0: "Directed", 1: "Balanced"})
    return pairs


def sample_non_edges(cos_df, rbo_edges, n=3000, seed=42):
    """Sample firm pairs not in the RBO edge set for null comparison."""
    np.random.seed(seed)
    edge_set = set(zip(rbo_edges["source"], rbo_edges["target"])) | \
               set(zip(rbo_edges["target"], rbo_edges["source"]))
    firms = cos_df.index.tolist()
    non_edge_sims = []
    attempts = 0
    while len(non_edge_sims) < n and attempts < n * 20:
        attempts += 1
        i, j = np.random.choice(len(firms), 2, replace=False)
        a, b = firms[i], firms[j]
        if (a, b) not in edge_set and (b, a) not in edge_set:
            non_edge_sims.append(cos_df.loc[a, b])
    return np.array(non_edge_sims)


# -- Statistical tests --------------------------------------------------------

def run_edge_type_comparison(pairs, non_edge_sims):
    """Test 1: Issue similarity across directed / balanced / non-edges."""
    directed = pairs[pairs["balanced"] == 0]["issue_cos"]
    balanced = pairs[pairs["balanced"] == 1]["issue_cos"]

    res = {}
    for label, vals in [("directed", directed), ("balanced", balanced),
                         ("non_edge", pd.Series(non_edge_sims))]:
        res[label] = {"mean": vals.mean(), "median": vals.median(),
                      "n": len(vals), "std": vals.std()}
    # Tests
    stat1, p1 = stats.mannwhitneyu(directed, balanced, alternative="two-sided")
    stat2, p2 = stats.mannwhitneyu(directed, non_edge_sims, alternative="greater")
    stat3, p3 = stats.mannwhitneyu(balanced, non_edge_sims, alternative="greater")
    return res, {"dir_vs_bal": p1, "dir_vs_noned": p2, "bal_vs_noned": p3}


def run_rbo_weight_correlation(pairs):
    """Test 2: Does issue similarity predict RBO weight?"""
    r_cos, p_cos = stats.spearmanr(pairs["issue_cos"], pairs["rbo_weight"])
    r_nt,  p_nt  = stats.spearmanr(pairs["issue_cos"],
                                    pairs["net_temporal"].abs())
    results = {"cos_vs_rbo_weight": (r_cos, p_cos),
               "cos_vs_abs_net_temporal": (r_nt, p_nt)}
    if pairs["issue_rbo"].notna().any():
        r_r, p_r = stats.spearmanr(pairs["issue_rbo"].dropna(),
                                    pairs.loc[pairs["issue_rbo"].notna(), "rbo_weight"])
        results["issue_rbo_vs_rbo_weight"] = (r_r, p_r)
    return results


def run_mediation_by_issue_overlap(pairs):
    """Test 3: Does high issue overlap suppress the need for a channel?"""
    directed = pairs[(pairs["balanced"] == 0) & (pairs["n_dir_bills"] > 0)].copy()
    q_lo = directed["issue_cos"].quantile(0.25)
    q_hi = directed["issue_cos"].quantile(0.75)
    lo   = directed[directed["issue_cos"] <= q_lo]
    hi   = directed[directed["issue_cos"] >= q_hi]
    return {
        "low_overlap":  {"n": len(lo), "med_rate": lo["any_mediated"].mean(),
                         "net_rate": lo["net_connected"].mean(),
                         "threshold": q_lo},
        "high_overlap": {"n": len(hi), "med_rate": hi["any_mediated"].mean(),
                         "net_rate": hi["net_connected"].mean(),
                         "threshold": q_hi},
    }


def run_community_comparison():
    """Test 4: Issue RBO vs directed influence community alignment (NMI/ARI)."""
    try:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        comm_rbo  = pd.read_csv(DATA_DIR / "communities" / "communities_rbo.csv")
        comm_iss  = DATA_DIR / "communities" / "communities_issue_rbo.csv"
        if not Path(comm_iss).exists():
            return None
        comm_iss  = pd.read_csv(comm_iss).rename(columns={"firm": "fortune_name",
                                                            "community": "comm_issue"})
        comm_rbo  = comm_rbo.rename(columns={"community_rbo": "comm_rbo"})
        merged = comm_rbo.merge(comm_iss, on="fortune_name", how="inner")
        nmi = normalized_mutual_info_score(merged["comm_rbo"], merged["comm_issue"])
        ari = adjusted_rand_score(merged["comm_rbo"], merged["comm_issue"])
        return {"n": len(merged), "nmi": nmi, "ari": ari}
    except Exception as e:
        return {"error": str(e)}


# -- Visualization ------------------------------------------------------------

def make_figure(pairs, non_edge_sims, edge_type_res, edge_pvals, corr_res, med_res):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: Issue similarity distributions by edge type ──────────────────
    ax = fig.add_subplot(gs[0, 0])
    directed = pairs[pairs["balanced"] == 0]["issue_cos"]
    balanced = pairs[pairs["balanced"] == 1]["issue_cos"]
    non_edge = pd.Series(non_edge_sims)

    bp = ax.boxplot(
        [non_edge, balanced, directed],
        labels=["Non-edges\n(random pairs)", "Balanced\nRBO edges", "Directed\nRBO edges"],
        patch_artist=True, widths=0.5, showfliers=False,
        medianprops=dict(color="white", linewidth=2),
    )
    cols = [GRAY, STEEL, GREEN]
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.8)

    # Significance brackets
    y_max = max(directed.quantile(0.9), balanced.quantile(0.9), non_edge.quantile(0.9))
    for i, (p, x1, x2) in enumerate([(edge_pvals["bal_vs_noned"], 1, 2),
                                       (edge_pvals["dir_vs_noned"], 1, 3)]):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        h = y_max * (1.08 + 0.06*i)
        ax.plot([x1, x1, x2, x2], [h, h*1.01, h*1.01, h], color="black", linewidth=1)
        ax.text((x1+x2)/2, h*1.015, sig, ha="center", fontsize=9)
    ax.set_ylabel("Issue-code cosine similarity")
    ax.set_title("A.  Issue Similarity by Edge Type\n(directed / balanced / non-edge pairs)")
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel B: RBO weight vs issue similarity scatter ───────────────────────
    ax = fig.add_subplot(gs[0, 1])
    dir_pairs = pairs[pairs["balanced"] == 0]
    ax.scatter(dir_pairs["issue_cos"], dir_pairs["rbo_weight"],
               s=18, alpha=0.35, color=STEEL)
    # Regression
    m, b, _, _, _ = stats.linregress(dir_pairs["issue_cos"], dir_pairs["rbo_weight"])
    xr = np.linspace(0, dir_pairs["issue_cos"].max(), 100)
    ax.plot(xr, m*xr + b, "--", color=AMBER, linewidth=2)
    r, p = corr_res["cos_vs_rbo_weight"]
    ax.text(0.97, 0.05, f"ρ = {r:.3f}  (p={p:.4f})",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=AMBER)
    ax.set_xlabel("Issue-code cosine similarity")
    ax.set_ylabel("RBO edge weight")
    ax.set_title("B.  Issue Similarity vs RBO Edge Weight\n(directed edges only)")
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel C: Mediation rate by issue overlap quartile ────────────────────
    ax = fig.add_subplot(gs[1, 0])
    dir_w_med = pairs[(pairs["balanced"] == 0) & (pairs["n_dir_bills"] > 0)].copy()
    dir_w_med["issue_quartile"] = pd.qcut(
        dir_w_med["issue_cos"], q=4,
        labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]
    )
    qt_stats = dir_w_med.groupby("issue_quartile", observed=True).agg(
        n=("any_mediated", "count"),
        bill_med_rate=("any_mediated", "mean"),
        net_conn_rate=("net_connected", "mean"),
    ).reset_index()
    x = np.arange(len(qt_stats))
    w = 0.35
    ax.bar(x - w/2, qt_stats["bill_med_rate"] * 100, w,
           label="Bill-level mediated", color=STEEL, alpha=0.85)
    ax.bar(x + w/2, qt_stats["net_conn_rate"] * 100, w,
           label="Network-connected", color=AMBER, alpha=0.85)
    for i, row in qt_stats.iterrows():
        ax.text(i, max(row["bill_med_rate"], row["net_conn_rate"]) * 100 + 0.02,
                f"n={row['n']}", ha="center", fontsize=8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(qt_stats["issue_quartile"], fontsize=9)
    ax.set_xlabel("Issue similarity quartile")
    ax.set_ylabel("Mediation rate (%)")
    ax.set_title("C.  Affiliation Mediation Rate by Issue Overlap\n"
                 "(directed edges with ≥1 directed bill)")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel D: Issue similarity CDF for directed vs balanced ───────────────
    ax = fig.add_subplot(gs[1, 1])
    for vals, label, col in [(directed, "Directed edges", GREEN),
                              (balanced, "Balanced edges", STEEL),
                              (non_edge, "Non-edges (sample)", GRAY)]:
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v)+1) / len(sorted_v)
        ax.plot(sorted_v, cdf, label=f"{label} (n={len(vals):,})", color=col, linewidth=2)
    ax.set_xlabel("Issue-code cosine similarity")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("D.  Issue Similarity CDF by Edge Type\n"
                 "(right-shifted CDF = higher similarity)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "Channel 3: Does Issue-Space Overlap Drive Directed Influence?\n"
        "116th Congress — Fortune 500 firms",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure -> {OUT_PNG.name}")


# -- Main ---------------------------------------------------------------------

def main():
    _orig = sys.stdout
    _f    = open(OUT_TXT, "w")
    sys.stdout = _Tee(_orig, _f)

    try:
        print("=" * 70)
        print("CHANNEL 3: ISSUE-SPACE CORRELATED RESPONSE AS DRIVER OF")
        print("           DIRECTED INFLUENCE")
        print("=" * 70)
        print("\nHypothesis: firms in the same regulatory domain independently")
        print("track and adopt the same bills. The RBO temporal precedence")
        print("reflects correlated monitoring speed, not information transmission.\n")

        print("Loading data...")
        rbo_edges = pd.read_csv(DATA_DIR / "rbo_directed_influence.csv")
        med_df    = pd.read_csv(DATA_DIR / "affiliation_mediated_adoption.csv")
        cos_df, pivot, firms, codes = build_issue_cosine_matrix()
        issue_rbo_sim = load_rbo_issue_edges()

        print(f"  RBO edges: {len(rbo_edges):,}  |  "
              f"Firms with issue data: {len(firms):,}  |  "
              f"Issue codes: {len(codes):,}")
        print(f"  Issue RBO edges loaded: {'yes' if issue_rbo_sim else 'no (run issue_rbo_similarity_network.py first)'}")

        pairs = build_pair_dataset(rbo_edges, cos_df, issue_rbo_sim, med_df)
        print(f"  Pairs in analysis: {len(pairs):,}  "
              f"(directed: {(pairs['balanced']==0).sum():,}  "
              f"balanced: {(pairs['balanced']==1).sum():,})")

        non_edge_sims = sample_non_edges(cos_df, rbo_edges)
        print(f"  Non-edge null sample: {len(non_edge_sims):,}")

        # Test 1
        print("\n" + "─" * 70)
        print("TEST 1: Issue Similarity Distribution by Edge Type")
        print("─" * 70)
        edge_type_res, edge_pvals = run_edge_type_comparison(pairs, non_edge_sims)
        print(f"\n  {'Group':<20}  {'n':>6}  {'Mean':>7}  {'Median':>7}  {'Std':>6}")
        for label, r in edge_type_res.items():
            print(f"  {label:<20}  {r['n']:>6,}  {r['mean']:>7.4f}  "
                  f"{r['median']:>7.4f}  {r['std']:>6.4f}")
        print(f"\n  Mann-Whitney p-values:")
        print(f"    Directed vs Balanced:  p = {edge_pvals['dir_vs_bal']:.4f}")
        print(f"    Directed vs Non-edge:  p = {edge_pvals['dir_vs_noned']:.4f}")
        print(f"    Balanced vs Non-edge:  p = {edge_pvals['bal_vs_noned']:.4f}")

        # Test 2
        print("\n" + "─" * 70)
        print("TEST 2: Issue Similarity as Predictor of RBO Edge Weight")
        print("─" * 70)
        corr_res = run_rbo_weight_correlation(pairs)
        for metric, (r, p) in corr_res.items():
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {metric:<35}: ρ = {r:>7.4f}  p = {p:.4f}  {sig}")

        # Test 3
        print("\n" + "─" * 70)
        print("TEST 3: Affiliation Mediation Rate by Issue Overlap Quartile")
        print("─" * 70)
        med_res = run_mediation_by_issue_overlap(pairs)
        print(f"\n  Low-overlap pairs  (≤Q25, cosine≤{med_res['low_overlap']['threshold']:.3f}), "
              f"n={med_res['low_overlap']['n']}:")
        print(f"    Bill-level mediated: {med_res['low_overlap']['med_rate']*100:.1f}%")
        print(f"    Network-connected:   {med_res['low_overlap']['net_rate']*100:.1f}%")
        print(f"\n  High-overlap pairs (≥Q75, cosine≥{med_res['high_overlap']['threshold']:.3f}), "
              f"n={med_res['high_overlap']['n']}:")
        print(f"    Bill-level mediated: {med_res['high_overlap']['med_rate']*100:.1f}%")
        print(f"    Network-connected:   {med_res['high_overlap']['net_rate']*100:.1f}%")

        # Test 4
        print("\n" + "─" * 70)
        print("TEST 4: Issue RBO Community vs Directed Influence Community (NMI/ARI)")
        print("─" * 70)
        comm_res = run_community_comparison()
        if comm_res and "error" not in comm_res:
            print(f"\n  n firms matched: {comm_res['n']}")
            print(f"  NMI (issue RBO vs directed influence communities): {comm_res['nmi']:.4f}")
            print(f"  ARI (issue RBO vs directed influence communities): {comm_res['ari']:.4f}")
            print(f"  (NMI=0 → independent; NMI=1 → identical partition)")
        elif comm_res and "error" in comm_res:
            print(f"  Error: {comm_res['error']}")
        else:
            print("  Skipped — issue RBO community file not available.")

        # Interpretation
        print("\n" + "─" * 70)
        print("INTERPRETATION")
        print("─" * 70)
        dir_mean = edge_type_res["directed"]["mean"]
        bal_mean = edge_type_res["balanced"]["mean"]
        non_mean = edge_type_res["non_edge"]["mean"]
        rho_w, p_w = corr_res["cos_vs_rbo_weight"]

        print(f"\n  Mean issue similarity:")
        print(f"    Directed edges: {dir_mean:.4f}  |  Balanced: {bal_mean:.4f}  "
              f"|  Non-edges: {non_mean:.4f}")
        print(f"\n  Issue similarity → RBO weight: ρ = {rho_w:.4f}  (p = {p_w:.4f})")
        print()
        if edge_pvals["dir_vs_noned"] < 0.05:
            print("  SUPPORT for correlated-response hypothesis: directed pairs have")
            print("  significantly higher issue similarity than random firm pairs,")
            print("  consistent with firms in the same regulatory domain independently")
            print("  tracking the same bills (Hojnacki 1997; Schlozman & Tierney 1986).")
        else:
            print("  NO SIGNIFICANT issue-overlap elevation for directed pairs over")
            print("  random pairs. The directed influence signal does not appear to be")
            print("  purely driven by domain-correlated monitoring.")
        if p_w < 0.05:
            print(f"\n  Issue similarity is a significant predictor of RBO edge weight")
            print(f"  (ρ = {rho_w:.3f}), meaning firms in the same policy domain not only")
            print(f"  form directed pairs — they are more strongly similar in their")
            print(f"  ranked bill portfolios. This is the strongest evidence for the")
            print(f"  correlated-response channel.")
        print()

        make_figure(pairs, non_edge_sims, edge_type_res, edge_pvals, corr_res, med_res)
        print(f"  Text output -> {OUT_TXT.name}")
        print("=" * 70)

    finally:
        sys.stdout = _orig
        _f.close()


if __name__ == "__main__":
    main()
