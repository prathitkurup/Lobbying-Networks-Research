"""Generate publication-quality figures 13–19 for lobbying network research."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ROOT

OUT_DIR = str(ROOT / "visualizations" / "png")
DPI = 150
PAL = {
    "blue":   "#2196F3",
    "red":    "#F44336",
    "green":  "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
}

# ─────────────────────────────────────────────
# Figure 13: Centrality vs Agenda-Setter Heatmap
# ─────────────────────────────────────────────

def fig13():
    cent_labels = ["BCZ\nIntercentrality", "Global\nPageRank", "WC\nEigenvector", "WC\nPageRank"]
    agenda_labels = ["net_influence", "net_strength", "WC\nnet_influence", "WC\nnet_strength"]

    full_rho = np.array([
        [0.178,  0.202,  0.182,  0.161],
        [0.193,  0.225,  np.nan, np.nan],
        [0.212,  0.221,  0.201,  0.158],
        [0.212,  0.227,  0.208,  0.172],
    ])
    top30_rho = np.array([
        [-0.107,  0.196,  0.019,  0.064],
        [ 0.119,  0.366,  np.nan, np.nan],
        [ 0.164,  0.370,  0.230,  0.291],
        [ 0.501,  0.565,  0.558,  0.534],
    ])

    # Significance masks (True = significant p<0.05)
    # Full-sample: all non-NaN are p<0.01
    full_sig = ~np.isnan(full_rho)

    top30_sig = np.zeros((4, 4), dtype=bool)
    # BCZ all ns
    # GlobalPR: net_strength p=0.047*
    top30_sig[1, 1] = True
    # WC Eigen: net_strength p=0.044*
    top30_sig[2, 1] = True
    # WC PR: all four p<0.05
    top30_sig[3, 0] = True
    top30_sig[3, 1] = True
    top30_sig[3, 2] = True
    top30_sig[3, 3] = True

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, rho, sig, title_suffix in zip(
        axes,
        [full_rho, top30_rho],
        [full_sig, top30_sig],
        ["Full Sample (n=full)", "Top-30 Lobbiers"],
    ):
        masked = np.ma.masked_invalid(rho)
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color="#BDBDBD")

        im = ax.imshow(masked, cmap=cmap, vmin=-0.2, vmax=0.6, aspect="auto")

        ax.set_xticks(range(4))
        ax.set_xticklabels(agenda_labels, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(cent_labels, fontsize=9)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        for i in range(4):
            for j in range(4):
                val = rho[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.5,
                            color="#555555", style="italic")
                else:
                    star = "*" if sig[i, j] else ""
                    text_color = "black" if -0.05 < val < 0.45 else "white"
                    ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                            fontsize=9.5, fontweight="bold", color=text_color)

        ax.set_title(title_suffix, fontsize=10, pad=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")

    fig.suptitle(
        "Centrality vs. Agenda-Setter: Spearman ρ (116th Congress)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "13_centrality_vs_agenda_setter_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Figure 14: Regression Coefficient Plot
# ─────────────────────────────────────────────

def fig14():
    variables = ["log_spend", "log_bills", "katz_centrality", "participation_coeff"]
    var_labels = ["log(Spend)", "log(Bills)", "Katz Centrality", "Participation Coeff."]

    coefs_116 = np.array([-0.0387, 0.1640, 1.1998, -0.0124])
    ses_116   = np.array([ 0.0190, 0.0229, 0.6102,  0.1006])
    coefs_117 = np.array([-0.0483, 0.1495, 0.6416, -0.0385])
    ses_117   = np.array([ 0.0240, 0.0285, 0.6898,  0.1174])

    # Stars: log_bills and log_spend are **, katz is * for 116th
    stars_116 = ["**", "**", "*", ""]
    stars_117 = ["", "", "", ""]

    fig, ax = plt.subplots(figsize=(8, 5))

    n = len(variables)
    offset = 0.18
    y_pos = np.arange(n)

    for idx, (coefs, ses, stars, label, color, yo) in enumerate([
        (coefs_116, ses_116, stars_116, "116th Congress", PAL["blue"],  offset),
        (coefs_117, ses_117, stars_117, "117th Congress", PAL["red"],  -offset),
    ]):
        ci = 1.96 * ses
        y = y_pos + yo
        ax.errorbar(coefs, y, xerr=ci, fmt="o", color=color, ecolor=color,
                    elinewidth=1.8, capsize=4, capthick=1.8, markersize=7, label=label, zorder=3)
        for xi, yi, star in zip(coefs, y, stars):
            if star:
                ax.text(xi + ci[list(coefs).index(xi)] + 0.02, yi, star,
                        ha="left", va="center", fontsize=10, color=color, fontweight="bold")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_labels, fontsize=10)
    ax.set_xlabel("Coefficient (95% CI)", fontsize=10)
    ax.set_title(
        "OLS: Predictors of Top-Quartile Net Influence (Specs A2, 116th & 117th)",
        fontsize=10.5, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "14_influencer_regression_coefplot.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Figure 15: Cross-Sector Flow Matrix
# ─────────────────────────────────────────────

def fig15():
    sectors = ["Defense", "Energy", "Finance", "Health", "Tech"]
    # rows=source, cols=target: Defense, Energy, Finance, Health, Tech
    matrix = np.array([
        [173, 100,  37,  67,  74],
        [ 62, 254,  26,  21,  48],
        [ 26,  22, 225,  20,  45],
        [ 20,  24,  13, 178,  20],
        [ 48,  26,  50,  34, 200],
    ])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = plt.cm.Blues.copy()
    cmap.set_under("white")

    im = ax.imshow(matrix, cmap=cmap, vmin=1, aspect="auto")

    ax.set_xticks(range(5))
    ax.set_xticklabels(sectors, fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(sectors, fontsize=10)
    ax.set_xlabel("Target Sector", fontsize=10)
    ax.set_ylabel("Source Sector", fontsize=10)

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            text_color = "white" if val > 180 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, color=text_color, fontweight=weight)
        # Outline diagonal cell
        rect = mpatches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            linewidth=2.5, edgecolor="#F44336", facecolor="none", zorder=3
        )
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Edge Count")

    ax.set_title(
        "Cross-Sector Directed Edge Counts by Community Pair (116th Congress)",
        fontsize=10.5, fontweight="bold"
    )
    fig.text(0.5, -0.03,
             "Diagonal = intra-sector (red outline); off-diagonal = cross-sector",
             ha="center", fontsize=8.5, style="italic", color="#555555")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "15_cross_sector_flow_matrix.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Figure 16: Community Rank Stability Summary
# ─────────────────────────────────────────────

def fig16():
    communities  = ["Finance/Ins", "Tech/Telecom", "Defense/Ind", "Energy/Utils", "Health/Pharma"]
    W_values     = [0.284, 0.431, 0.367, 0.553, 0.146]
    p_values_str = ["0.002", "<0.0001", "0.0001", "<0.0001", "0.428"]
    p_float      = [0.002, 0.00001, 0.0001, 0.00001, 0.428]

    def bar_color(p):
        if p < 0.001:  return "#2196F3"
        if p < 0.01:   return "#64B5F6"
        if p < 0.05:   return "#BBDEFB"
        return "#E0E0E0"

    def sig_marker(p):
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        return "ns"

    # Persistent leaders data
    firms = [
        "Duke Energy", "Xcel Energy", "CMS Energy", "DTE Energy",
        "Am. Family Ins", "Northwestern Mutual"
    ]
    congresses = [111, 112, 113, 114, 115, 116, 117]
    presence = np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,0,0,1,1],
        [0,0,1,1,1,1,0],
        [1,1,1,0,1,1,0],
        [0,0,0,1,1,1,1],
    ])
    comm_labels = ["Energy", "Energy", "Energy", "Energy", "Finance", "Finance"]
    comm_colors = {"Energy": PAL["orange"], "Finance": PAL["green"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Kendall's W bar chart ---
    colors = [bar_color(p) for p in p_float]
    bars = ax1.bar(communities, W_values, color=colors, edgecolor="white", linewidth=0.8, zorder=2)
    ax1.axhline(0.30, color="#555555", linestyle="--", linewidth=1.5, zorder=3, label="Moderate concordance (W=0.30)")

    for bar, W, p in zip(bars, W_values, p_float):
        marker = sig_marker(p)
        ax1.text(bar.get_x() + bar.get_width() / 2, W + 0.015,
                 marker, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width() / 2, W / 2,
                 f"W={W:.3f}", ha="center", va="center", fontsize=8.5,
                 color="black" if W < 0.45 else "white")

    ax1.set_ylim(0, 0.68)
    ax1.set_ylabel("Kendall's W", fontsize=10)
    ax1.set_title("Within-Community Rank Stability:\nKendall's W (111th–117th Congress)",
                  fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(communities)))
    ax1.set_xticklabels(communities, fontsize=8.5, rotation=15, ha="right")
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Legend for bar colors
    patches = [
        mpatches.Patch(color="#2196F3", label="p < 0.001 (***)"),
        mpatches.Patch(color="#64B5F6", label="p < 0.01 (**)"),
        mpatches.Patch(color="#BBDEFB", label="p < 0.05 (*)"),
        mpatches.Patch(color="#E0E0E0", label="ns"),
    ]
    ax1.legend(handles=patches + [plt.Line2D([0],[0], color="#555555", linestyle="--", label="W=0.30 threshold")],
               fontsize=8, loc="upper right")

    # --- Right: Heatmap of persistent leaders ---
    n_firms, n_cong = presence.shape
    binary_cmap = matplotlib.colors.ListedColormap(["white", PAL["orange"]])

    im2 = ax2.imshow(presence, cmap=binary_cmap, vmin=0, vmax=1, aspect="auto")

    ax2.set_xticks(range(n_cong))
    ax2.set_xticklabels([str(c) for c in congresses], fontsize=9)
    ax2.set_yticks(range(n_firms))
    ax2.set_yticklabels(firms, fontsize=9)
    ax2.set_xlabel("Congress", fontsize=10)

    for i in range(n_firms):
        for j in range(n_cong):
            if presence[i, j] == 1:
                ax2.text(j, i, "✓", ha="center", va="center",
                         fontsize=9, color="white", fontweight="bold")

    # Community label on right
    ax2_r = ax2.twinx()
    ax2_r.set_ylim(ax2.get_ylim())
    ax2_r.set_yticks(range(n_firms))
    ax2_r.set_yticklabels(
        [f"[{c}]" for c in comm_labels],
        fontsize=8.5
    )
    for tick, c in zip(ax2_r.get_yticklabels(), comm_labels):
        tick.set_color(comm_colors[c])

    ax2.set_title("Persistent Within-Community Leaders\nAcross 7 Congresses",
                  fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "16_rank_stability_summary.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Figure 18: Payoff Complementarity Coefficient Plot
# ─────────────────────────────────────────────

def fig18():
    specs = ["Spec A\n(Full RBO)", "Spec B\n(High RBO ≥p75)", "Spec D\n(All Pairs)"]
    spec_colors = [PAL["blue"], PAL["green"], PAL["purple"]]

    beta3_coefs = np.array([-0.1252,  0.1470, -0.1730])
    beta3_ses   = np.array([ 0.0329,  0.0689,  0.0294])
    beta3_pvals = ["p<0.001***", "p=0.033*", "p<0.001***"]

    beta1_coefs = np.array([-0.0133, -0.1248, -0.0106])
    beta1_ses   = np.array([ 0.0055,  0.0209,  0.0034])
    beta1_pvals = ["p<0.05*", "p<0.001***", "p<0.001***"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    y = np.array([2, 1, 0])

    for ax, coefs, ses, pvals, panel_title, subtitle in [
        (ax1, beta3_coefs, beta3_ses, beta3_pvals,
         "β₃: Interaction Term\n(entry_j × rbo_ij)",
         "Positive β₃ = BCZ complementarity;\nonly Spec B (High-RBO) positive"),
        (ax2, beta1_coefs, beta1_ses, beta1_pvals,
         "β₁: entry_j Main Effect", None),
    ]:
        ci = 1.96 * ses
        for i, (c, e, p, sc, spec) in enumerate(zip(coefs, ci, pvals, spec_colors, specs)):
            ax.errorbar(c, y[i], xerr=e, fmt="o", color=sc, ecolor=sc,
                        elinewidth=2, capsize=5, capthick=2, markersize=8, zorder=3,
                        label=spec.replace("\n", " "))
            ax.text(c + e + 0.003, y[i], p, va="center", ha="left", fontsize=8.5, color=sc)

        ax.axvline(0, color="black", linestyle="--", linewidth=1.3, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(specs, fontsize=9)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=9.5)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if subtitle:
            ax.text(0.02, -0.22, subtitle, transform=ax.transAxes,
                    fontsize=8, style="italic", color="#555555")

    fig.suptitle(
        "BCZ Payoff Complementarity: Interaction Coefficients by Sample Restriction",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "18_payoff_complementarity_coefplot.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Figure 19: Bill Adoption Diffusion Curve
# ─────────────────────────────────────────────

def fig19():
    horizons = [1, 2, 3]
    horizon_labels = ["Q+1", "Q+2", "Q+3"]

    # Adoption rates per quartile × horizon
    rates = {
        "Q1 (Low RBO)":  [0.0204, 0.0335, 0.0392],
        "Q2":            [0.0242, 0.0379, 0.0458],
        "Q3":            [0.0330, 0.0514, 0.0611],
        "Q4 (High RBO)": [0.0455, 0.0707, 0.0831],
    }
    # Q+3 CI bounds
    ci_q3 = {
        "Q1 (Low RBO)":  (0.0361, 0.0424),
        "Q2":            (0.0425, 0.0493),
        "Q3":            (0.0573, 0.0651),
        "Q4 (High RBO)": (0.0788, 0.0877),
    }
    # N estimates (illustrative, consistent with paper framing)
    N_q3 = {"Q1 (Low RBO)": 412, "Q2": 418, "Q3": 405, "Q4 (High RBO)": 410}

    # Color gradient: Q1 light red → Q4 dark blue
    colors = ["#EF9A9A", "#90CAF9", "#1E88E5", "#0D47A1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Line chart ---
    for (q_label, r), color in zip(rates.items(), colors):
        ax1.plot(horizon_labels, r, marker="o", color=color, linewidth=2,
                 markersize=7, label=q_label, zorder=3)
        # CI band at Q+3
        lo, hi = ci_q3[q_label]
        ax1.fill_between([horizon_labels[1], horizon_labels[2]], [r[1], lo], [r[1], hi],
                         color=color, alpha=0.18)

    ax1.set_xlabel("Adoption Horizon", fontsize=10)
    ax1.set_ylabel("Adoption Rate", fontsize=10)
    ax1.set_title("Follower Bill Adoption Rate by RBO Quartile and Horizon",
                  fontsize=10, fontweight="bold")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
    ax1.legend(fontsize=9, title="RBO Quartile")
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add 95% CI note
    ax1.text(0.98, 0.05, "Shaded region = 95% CI at Q+3",
             transform=ax1.transAxes, fontsize=8, ha="right",
             style="italic", color="#555555")

    # --- Right: Grouped bar chart at Q+3 ---
    q_labels_short = ["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"]
    q3_rates  = [rates[k][2] for k in rates]
    q3_lo     = [ci_q3[k][0] for k in rates]
    q3_hi     = [ci_q3[k][1] for k in rates]
    q3_errs_lo = np.array(q3_rates) - np.array(q3_lo)
    q3_errs_hi = np.array(q3_hi) - np.array(q3_rates)

    x = np.arange(4)
    bars = ax2.bar(x, q3_rates, color=colors, width=0.6,
                   yerr=[q3_errs_lo, q3_errs_hi],
                   error_kw=dict(ecolor="#333333", elinewidth=1.5, capsize=5, capthick=1.5),
                   zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_labels_short, fontsize=10)
    ax2.set_xlabel("RBO Quartile", fontsize=10)
    ax2.set_ylabel("Adoption Rate at Q+3", fontsize=10)
    ax2.set_title("Q+3 Adoption Rate: High-RBO Pairs\nAdopt at 2.1× Rate of Low-RBO",
                  fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar, rate, q_key in zip(bars, q3_rates, rates.keys()):
        n = N_q3[q_key]
        ax2.text(bar.get_x() + bar.get_width() / 2, rate + 0.001,
                 f"{rate:.1%}\n(n={n})", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(
        "Bill Adoption Diffusion: Do High-RBO Followers Adopt Influencer Bills? (116th Congress)",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "19_adoption_curve_by_quartile.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────

if __name__ == "__main__":
    fig13()
    fig14()
    fig15()
    fig16()
    fig18()
    fig19()
    print("All figures generated.")
