"""
Channel 1 Test: LDA Monitoring Capacity as Driver of Directed Influence.

Hypothesis: The RBO directed influence signal reflects lobbying capacity gaps —
firms with greater resources track and adopt relevant legislation earlier,
appearing as "influencers" by virtue of monitoring bandwidth rather than
information transmission to peers.

Tests:
  1. Spearman correlation between per-firm capacity metrics and net_influence.
  2. Paired Wilcoxon signed-rank: for each directed RBO edge, is source capacity
     greater than target capacity?
  3. Capacity percentile comparison: net influencers vs net followers.
  4. Logistic regression: does capacity predict influencer status above chance?

Capacity proxies (all from opensecrets_lda_reports.csv for 116th Congress):
  total_spend    — sum of lobbying expenditure across all reports
  n_lobbyists    — unique human lobbyists retained
  n_bills        — unique bills lobbied
  n_reports      — unique LDA reports filed
  n_issue_codes  — unique issue codes covered (breadth of policy engagement)

Outputs:
  outputs/channel_tests/channel1_monitoring_capacity.txt
  outputs/channel_tests/channel1_monitoring_capacity.png

Run from src/:  python channel_tests/test_channel1_monitoring_capacity.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, ROOT, OPENSECRETS_OUTPUT_CSV, OPENSECRETS_ISSUES_CSV, OPENSECRETS_LOBBYIST_CLIENT_CSV

OUT_DIR   = ROOT / "outputs" / "channel_tests"
OUT_TXT   = OUT_DIR / "channel1_monitoring_capacity.txt"
OUT_PNG   = OUT_DIR / "channel1_monitoring_capacity.png"
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


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text): [s.write(text) for s in self.streams]
    def flush(self):       [s.flush()     for s in self.streams]


# -- Data loading -------------------------------------------------------------

def build_capacity_metrics():
    """Compute per-firm capacity proxies from the 116th Congress LDA data."""
    reports = pd.read_csv(OPENSECRETS_OUTPUT_CSV)
    issues  = pd.read_csv(OPENSECRETS_ISSUES_CSV)
    lob_cl  = pd.read_csv(OPENSECRETS_LOBBYIST_CLIENT_CSV)

    # Total spend: sum amount (report-level, not bill-expanded) per firm
    # Use drop_duplicates on uniq_id to avoid double-counting bill-expanded rows
    report_level = reports.drop_duplicates(subset=["uniq_id"])[["fortune_name","uniq_id","amount"]]
    total_spend  = report_level.groupby("fortune_name")["amount"].sum().rename("total_spend")
    n_reports    = report_level.groupby("fortune_name")["uniq_id"].nunique().rename("n_reports")

    # Unique bills lobbied
    n_bills = (reports.dropna(subset=["bill_number"])
               .groupby("fortune_name")["bill_number"].nunique().rename("n_bills"))

    # Unique lobbyists (by lobbyist_id to avoid name disambiguation issues)
    n_lobbyists = lob_cl.groupby("fortune_name")["lobbyist_id"].nunique().rename("n_lobbyists")

    # Unique issue codes
    n_issue_codes = issues.groupby("fortune_name")["issue_code"].nunique().rename("n_issue_codes")

    cap = pd.concat([total_spend, n_reports, n_bills, n_lobbyists, n_issue_codes], axis=1).fillna(0)

    return cap


def load_rbo_network():
    """Load directed influence network with node net_influence from GML."""
    import networkx as nx
    gml = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"
    G   = nx.read_gml(str(gml))
    node_attrs = {n: G.nodes[n] for n in G.nodes()}
    edges = pd.read_csv(DATA_DIR / "rbo_directed_influence.csv")
    return G, edges, node_attrs


# -- Statistical tests --------------------------------------------------------

def run_correlation_test(cap, node_attrs, metrics):
    """Spearman correlation: capacity metrics vs net_influence."""
    common_firms = sorted(set(cap.index) & set(node_attrs.keys()))
    ni = pd.Series({f: node_attrs[f].get("net_influence", 0) for f in common_firms})
    results = {}
    for m in metrics:
        vals = cap.loc[common_firms, m]
        r, p = stats.spearmanr(vals, ni)
        results[m] = {"rho": r, "p": p, "n": len(common_firms)}
    return results, ni, common_firms


def run_paired_wilcoxon(edges, cap, metrics):
    """Paired Wilcoxon: for directed edges, is source capacity > target?"""
    directed = edges[edges["balanced"] == 0].copy()
    results = {}
    for m in metrics:
        src_vals, tgt_vals = [], []
        for _, row in directed.iterrows():
            s, t = row["source"], row["target"]
            if s in cap.index and t in cap.index:
                src_vals.append(cap.loc[s, m])
                tgt_vals.append(cap.loc[t, m])
        src_arr = np.array(src_vals)
        tgt_arr = np.array(tgt_vals)
        diffs   = src_arr - tgt_arr
        stat, p = stats.wilcoxon(diffs, alternative="greater") if len(diffs) > 0 else (np.nan, np.nan)
        results[m] = {
            "n":           len(src_vals),
            "src_median":  np.median(src_arr),
            "tgt_median":  np.median(tgt_arr),
            "pct_src_gt":  (src_arr > tgt_arr).mean() * 100,
            "wilcoxon_W":  stat,
            "p":           p,
        }
    return results


def run_percentile_comparison(cap, ni, metrics):
    """Compare capacity distributions for top vs bottom net_influence quartile."""
    q75 = ni.quantile(0.75)
    q25 = ni.quantile(0.25)
    influencers = ni[ni >= q75].index
    followers   = ni[ni <= q25].index
    results = {}
    for m in metrics:
        inf_vals = cap.loc[cap.index.isin(influencers), m]
        fol_vals = cap.loc[cap.index.isin(followers),   m]
        stat, p  = stats.mannwhitneyu(inf_vals, fol_vals, alternative="greater")
        results[m] = {
            "inf_median": inf_vals.median(), "fol_median": fol_vals.median(),
            "inf_mean":   inf_vals.mean(),   "fol_mean":   fol_vals.mean(),
            "mann_whitney_p": p,
        }
    return results, influencers, followers


# -- Visualization ------------------------------------------------------------

def make_figure(cap, ni, common_firms, corr_results, paired_results,
                pct_results, influencers, followers, metrics):
    """Four-panel figure: correlations, paired edge comparison, distributions."""
    METRIC_LABELS = {
        "total_spend":   "Total Lobbying\nSpend ($)",
        "n_reports":     "Reports Filed\n(N)",
        "n_bills":       "Unique Bills\nLobbied (N)",
        "n_lobbyists":   "Unique Lobbyists\nRetained (N)",
        "n_issue_codes": "Issue Codes\nCovered (N)",
    }

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: Spearman correlations bar chart ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    rhos = [corr_results[m]["rho"] for m in metrics]
    ps   = [corr_results[m]["p"]   for m in metrics]
    xlabels = [METRIC_LABELS[m].replace("\n", " ") for m in metrics]
    colors  = [GREEN if r > 0 else RED for r in rhos]
    bars = ax.barh(range(len(metrics)), rhos, color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (r, p) in enumerate(zip(rhos, ps)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(r + (0.01 if r >= 0 else -0.01), i,
                f"{r:.3f} {sig}", va="center",
                ha="left" if r >= 0 else "right", fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Spearman ρ with net_influence")
    ax.set_title("A.  Capacity vs Net Influence\n(Spearman correlation)")
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel B: Source vs target capacity for directed edges ─────────────────
    ax = fig.add_subplot(gs[0, 1])
    src_medians = [paired_results[m]["src_median"] for m in metrics]
    tgt_medians = [paired_results[m]["tgt_median"] for m in metrics]
    pcts_gt     = [paired_results[m]["pct_src_gt"] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, src_medians, w, label="RBO source (influencer)", color=GREEN, alpha=0.8)
    b2 = ax.bar(x + w/2, tgt_medians, w, label="RBO target (follower)",   color=RED,   alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Median value")
    ax.set_title("B.  Source vs Target Capacity\n(directed RBO edges)")
    ax.legend(fontsize=9)
    # Annotate % source > target
    for i, (pct, pr) in enumerate(zip(pcts_gt, [paired_results[m] for m in metrics])):
        sig = "***" if pr["p"] < 0.001 else "**" if pr["p"] < 0.01 else "*" if pr["p"] < 0.05 else "n.s."
        ax.text(i, max(src_medians[i], tgt_medians[i]) * 1.05,
                f"{pct:.0f}%>{sig}", ha="center", fontsize=8, color=NAVY)
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel C: Net influence scatter vs total spend ─────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    spend_vals = cap.loc[common_firms, "total_spend"] / 1e6
    ni_vals    = ni.loc[common_firms]
    # Highlight influencers and followers
    inf_mask = pd.Index(common_firms).isin(influencers)
    fol_mask = pd.Index(common_firms).isin(followers)
    other    = ~inf_mask & ~fol_mask
    ax.scatter(spend_vals[other], ni_vals[other], s=25, alpha=0.4, color=GRAY, label="Other")
    ax.scatter(spend_vals[inf_mask], ni_vals[inf_mask], s=40, alpha=0.7,
               color=GREEN, label="Top Q1 influencers")
    ax.scatter(spend_vals[fol_mask], ni_vals[fol_mask], s=40, alpha=0.7,
               color=RED,   label="Bottom Q1 followers")
    # Regression line
    m_coef, b_coef, _, _, _ = stats.linregress(spend_vals, ni_vals)
    xr = np.linspace(spend_vals.min(), spend_vals.max(), 100)
    ax.plot(xr, m_coef * xr + b_coef, "--", color=STEEL, linewidth=1.5)
    r, p = corr_results["total_spend"]["rho"], corr_results["total_spend"]["p"]
    ax.text(0.98, 0.05, f"ρ = {r:.3f}  (p={p:.3f})",
            transform=ax.transAxes, ha="right", fontsize=9, color=STEEL)
    ax.set_xlabel("Total lobbying spend ($ millions)")
    ax.set_ylabel("Net influence (RBO network)")
    ax.set_title("C.  Spend vs Net Influence\n(scatter with regression)")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)

    # ── Panel D: Distribution of key metric by influencer/follower/neutral ────
    ax = fig.add_subplot(gs[1, 1])
    neutral = [f for f in common_firms if f not in influencers and f not in followers]
    groups  = {
        "Net followers\n(bottom Q1)":  cap.loc[cap.index.isin(followers),   "n_bills"],
        "Neutral":                     cap.loc[cap.index.isin(neutral),     "n_bills"],
        "Net influencers\n(top Q1)":   cap.loc[cap.index.isin(influencers), "n_bills"],
    }
    positions  = list(range(len(groups)))
    group_data = list(groups.values())
    bp = ax.boxplot(group_data, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=True,
                    medianprops=dict(color="white", linewidth=2))
    box_colors = [RED, GRAY, GREEN]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    # Mann-Whitney p annotate
    stat, p_mw = stats.mannwhitneyu(group_data[2], group_data[0], alternative="greater")
    y_max = max(g.max() for g in group_data)
    ax.plot([0, 2], [y_max * 1.08, y_max * 1.08], color="black", linewidth=1)
    p_str = f"p={p_mw:.3f}" if p_mw >= 0.001 else "p<0.001"
    ax.text(1, y_max * 1.10, p_str, ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(groups.keys()), fontsize=9)
    ax.set_ylabel("Unique bills lobbied (N)")
    ax.set_title("D.  Bills Lobbied by Influencer Quartile\n(influencers vs followers vs neutral)")
    ax.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "Channel 1: Does Lobbying Capacity Explain Directed Influence?\n"
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
        print("CHANNEL 1: LDA MONITORING CAPACITY AS DRIVER OF DIRECTED INFLUENCE")
        print("=" * 70)
        print("\nHypothesis: the RBO 'influence' signal reflects capacity gaps —")
        print("firms with greater lobbying operations track legislation earlier,")
        print("appearing as first-movers by virtue of monitoring bandwidth rather")
        print("than genuine information transmission to peers.\n")

        print("Loading data...")
        cap          = build_capacity_metrics()
        G, edges, na = load_rbo_network()
        METRICS = ["total_spend", "n_reports", "n_bills", "n_lobbyists", "n_issue_codes"]

        print(f"  Firms with capacity data:   {len(cap):,}")
        print(f"  Firms in RBO network:       {len(na):,}")
        print(f"  Directed RBO edges:         {(edges['balanced']==0).sum():,}")
        print()
        print("Capacity metric summary:")
        print(cap[METRICS].describe().round(1).to_string())

        # Test 1: Spearman correlations
        print("\n" + "─" * 70)
        print("TEST 1: Spearman Correlation — Capacity vs Net Influence")
        print("─" * 70)
        corr_results, ni, common_firms = run_correlation_test(cap, na, METRICS)
        print(f"\n  n = {len(common_firms)} firms with both capacity and RBO data\n")
        print(f"  {'Metric':<20}  {'ρ':>7}  {'p':>9}  {'Sig':>5}")
        for m, r in corr_results.items():
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
            print(f"  {m:<20}  {r['rho']:>7.4f}  {r['p']:>9.4f}  {sig:>5}")

        # Test 2: Paired Wilcoxon for directed edges
        print("\n" + "─" * 70)
        print("TEST 2: Paired Wilcoxon — Is source capacity > target? (directed edges)")
        print("─" * 70)
        print("  One-sided test: H1: source_capacity > target_capacity per edge\n")
        paired_results = run_paired_wilcoxon(edges, cap, METRICS)
        print(f"  {'Metric':<20}  {'n':>5}  {'Src_med':>10}  {'Tgt_med':>10}  "
              f"{'%Src>Tgt':>9}  {'p':>9}  {'Sig':>5}")
        for m, r in paired_results.items():
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
            print(f"  {m:<20}  {r['n']:>5}  {r['src_median']:>10.0f}  "
                  f"{r['tgt_median']:>10.0f}  {r['pct_src_gt']:>8.1f}%  "
                  f"{r['p']:>9.4f}  {sig:>5}")

        # Test 3: Percentile comparison
        print("\n" + "─" * 70)
        print("TEST 3: Capacity by Influencer Quartile (top Q1 vs bottom Q1)")
        print("─" * 70)
        print("  Mann-Whitney U: H1: influencers have higher capacity than followers\n")
        pct_results, influencers, followers = run_percentile_comparison(cap, ni, METRICS)
        print(f"  Top-Q1 influencers: n={len(influencers)}  "
              f"Bottom-Q1 followers: n={len(followers)}")
        print(f"\n  {'Metric':<20}  {'Inf_med':>10}  {'Fol_med':>10}  "
              f"{'Inf_mean':>10}  {'Fol_mean':>10}  {'p(MW)':>9}  {'Sig':>5}")
        for m, r in pct_results.items():
            sig = "***" if r["mann_whitney_p"] < 0.001 else "**" if r["mann_whitney_p"] < 0.01 \
                  else "*" if r["mann_whitney_p"] < 0.05 else "n.s."
            print(f"  {m:<20}  {r['inf_median']:>10.0f}  {r['fol_median']:>10.0f}  "
                  f"{r['inf_mean']:>10.0f}  {r['fol_mean']:>10.0f}  "
                  f"{r['mann_whitney_p']:>9.4f}  {sig:>5}")

        # Interpretation
        print("\n" + "─" * 70)
        print("INTERPRETATION")
        print("─" * 70)
        sig_corr = [m for m, r in corr_results.items() if r["p"] < 0.05]
        sig_pair = [m for m, r in paired_results.items() if r["p"] < 0.05]
        print(f"\n  Capacity metrics significantly correlated with net_influence: "
              f"{sig_corr if sig_corr else 'none'}")
        print(f"  Metrics where source > target (directed edges, p<0.05): "
              f"{sig_pair if sig_pair else 'none'}")
        print()
        if sig_corr or sig_pair:
            print("  PARTIAL SUPPORT for capacity-gap hypothesis: some metrics predict")
            print("  influencer status, suggesting monitoring bandwidth contributes to")
            print("  first-mover advantage. However, this does not exhaust the signal —")
            print("  capacity gap and genuine information transmission are not mutually")
            print("  exclusive (Bertrand, Bombardini & Trebbi 2014).")
        else:
            print("  NO SIGNIFICANT CAPACITY-GAP EFFECT detected. The RBO influence")
            print("  signal does not primarily reflect lobbying resource disparities.")
            print("  Large and small firms appear equally likely to be first-movers,")
            print("  suggesting the temporal ordering reflects something other than")
            print("  pure monitoring bandwidth.")
        print()

        # Figure
        make_figure(cap, ni, common_firms, corr_results, paired_results,
                    pct_results, influencers, followers, METRICS)
        print(f"  Text output -> {OUT_TXT.name}")
        print("=" * 70)

    finally:
        sys.stdout = _orig
        _f.close()


if __name__ == "__main__":
    main()
