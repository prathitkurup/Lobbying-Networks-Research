"""
Visualization suite for affiliation-mediated adoption analysis.

Produces three figures saved to visualizations/png/:
  affiliation_mediation_summary.png   — four-panel: sparsity breakdown, channel
                                        comparison, lag distribution, alignment test
  affiliation_mediation_network.png   — RBO directed influence network with mediated
                                        edges and nodes highlighted
  affiliation_mediation_timeline.png  — bill-level adoption timeline for the 7
                                        affiliation-mediated pairs

Run after affiliation_mediated_adoption.py (reads its CSV outputs).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import networkx as nx
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ROOT

VIZ_DIR     = ROOT / "visualizations" / "png"
GML_PATH    = ROOT / "visualizations" / "gml" / "rbo_directed_influence.gml"
MED_CSV     = DATA_DIR / "affiliation_mediated_adoption.csv"
ENR_CSV     = DATA_DIR / "rbo_edges_enriched.csv"

VIZ_DIR.mkdir(parents=True, exist_ok=True)

# -- Style constants ----------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
})

NAVY    = "#1F3864"
STEEL   = "#2E5FA3"
AMBER   = "#E67E22"
EMERALD = "#27AE60"
CRIMSON = "#C0392B"
SLATE   = "#7F8C8D"
LGRAY   = "#ECF0F1"
MGRAY   = "#BDC3C7"

# Channel palette (colorblind-friendly: blue / orange / teal)
C_LOB_ONLY  = "#4C72B0"   # blue
C_FIRM_ONLY = "#DD8452"   # orange
C_BOTH      = "#55A868"   # green
C_NONE      = LGRAY

# Mediated edge colors for the network figure
C_MED_LOB  = AMBER
C_MED_FIRM = STEEL
C_UNMED    = "#CCCCCC"

QUARTER_LABELS = {1:"2019 Q1", 2:"2019 Q2", 3:"2019 Q3", 4:"2019 Q4",
                  5:"2020 Q1", 6:"2020 Q2", 7:"2020 Q3", 8:"2020 Q4"}


# -- Data loading -------------------------------------------------------------

def load_data():
    """Load mediation and enriched edge DataFrames."""
    med = pd.read_csv(MED_CSV)
    enr = pd.read_csv(ENR_CSV)
    return med, enr


# ── Figure 1: Four-panel summary ─────────────────────────────────────────────

def fig_summary(med):
    """Four-panel summary of mediation rates, lag, and alignment."""
    directed = med[med["is_bill_directed"]]
    n = len(directed)

    lob_only  = (directed["is_lobbyist_mediated"] & ~directed["is_firm_mediated"]).sum()
    firm_only = (directed["is_firm_mediated"]     & ~directed["is_lobbyist_mediated"]).sum()
    both      = (directed["is_lobbyist_mediated"] &  directed["is_firm_mediated"]).sum()
    none_n    = (~directed["is_any_mediated"]).sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Affiliation-Mediated Bill Adoption: Empirical Overview\n"
        "116th Congress (2019\u20132020), Fortune 500 firms",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Panel A: Mediation composition ───────────────────────────────────────
    ax = axes[0, 0]
    segments = [none_n, lob_only, firm_only, both]
    colors   = [C_NONE, C_LOB_ONLY, C_FIRM_ONLY, C_BOTH]
    labels   = ["Neither\n(unmediated)", "Lobbyist\nonly", "Firm\nonly", "Both\nchannels"]
    left = 0
    for val, col, lbl in zip(segments, colors, labels):
        bar = ax.barh(0, val, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.5)
        pct = 100 * val / n
        if val > 5:
            ax.text(left + val / 2, 0, f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if col not in (C_NONE, LGRAY) else "#444444")
        left += val

    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel("Directed bill-adoption pairs")
    ax.set_title("A.  Mediation Breakdown\n(directed pairs, lag > 0)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    legend_patches = [mpatches.Patch(facecolor=c, label=l, edgecolor="white")
                      for c, l in zip(colors, labels)]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, ncol=2,
              framealpha=0.9)

    # ── Panel B: Bill-level vs network-level rates ────────────────────────────
    ax = axes[0, 1]
    channels    = ["Lobbyist", "Firm\n(K-street only)", "Any channel"]
    bill_rates  = [
        100 * directed["is_lobbyist_mediated"].mean(),
        100 * directed["is_firm_mediated"].mean(),
        100 * directed["is_any_mediated"].mean(),
    ]
    net_rates = [
        100 * directed["net_lob_connected"].mean(),
        100 * directed["net_firm_connected"].mean(),
        100 * directed["net_any_connected"].mean(),
    ]
    x = np.arange(len(channels))
    w = 0.35
    b1 = ax.bar(x - w/2, bill_rates, w, label="Bill-level\n(same bill, first-quarter)",
                color=STEEL, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, net_rates,  w, label="Network-level\n(any shared intermediary)",
                color=AMBER, alpha=0.85, edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontsize=10)
    ax.set_ylabel("% of directed bill pairs")
    ax.set_title("B.  Bill-level vs Network-level Affiliation\n(of 3,184 directed bill pairs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(net_rates) * 1.5)

    # ── Panel C: Lag distribution by network connectivity ────────────────────
    ax = axes[1, 0]
    conn    = directed[directed["net_any_connected"]]["lag_quarters"]
    nonconn = directed[~directed["net_any_connected"]]["lag_quarters"]

    # Jittered strip + box
    jitter_scale = 0.08
    np.random.seed(42)
    jit_conn    = np.random.uniform(-jitter_scale, jitter_scale, len(conn))
    jit_nonconn = np.random.uniform(-jitter_scale, jitter_scale, len(nonconn))

    ax.scatter(conn    + jit_conn,    np.ones(len(conn))    * 1, alpha=0.5, s=30,
               color=AMBER,   zorder=3, label=f"Network-connected (n={len(conn)})")
    ax.scatter(nonconn + jit_nonconn, np.ones(len(nonconn)) * 0, alpha=0.08, s=10,
               color=STEEL,   zorder=2, label=f"Not connected (n={len(nonconn):,})")

    # Mean lines
    ax.axhline(1, xmin=0, xmax=0, color=AMBER)  # legend only
    ax.axvline(conn.mean(),    color=AMBER, linestyle="--", linewidth=1.8,
               label=f"Connected mean: {conn.mean():.2f} q")
    ax.axvline(nonconn.mean(), color=STEEL, linestyle="--", linewidth=1.8,
               label=f"Non-connected mean: {nonconn.mean():.2f} q")

    # Mann-Whitney p-value annotation
    _, pval = stats.mannwhitneyu(conn, nonconn, alternative="less")
    ax.text(0.98, 0.96, f"Mann-Whitney p = {pval:.3f} (n.s.)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color="#555555", style="italic")

    ax.set_xlabel("Adoption lag (quarters)")
    ax.set_ylabel("Group (0 = not connected, 1 = connected)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not\nconnected", "Network-\nconnected"], fontsize=10)
    ax.set_title("C.  Adoption Lag: Network-connected vs Non-connected\n"
                 "(network-level affiliation; directed pairs only)")
    ax.set_xlim(0, directed["lag_quarters"].max() + 0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel D: Alignment test ───────────────────────────────────────────────
    ax = axes[1, 1]
    d0 = med[(med["rbo_balanced"] == 0) & med["is_bill_directed"]].copy()
    d0["align"] = d0["leader"] == d0["rbo_source"]

    groups = [
        ("All directed", d0),
        ("Non-mediated", d0[~d0["is_any_mediated"]]),
        ("Lobbyist-\nmediated", d0[d0["is_lobbyist_mediated"]]),
        ("Any-\nmediated", d0[d0["is_any_mediated"]]),
    ]
    bar_colors = [SLATE, MGRAY, C_LOB_ONLY, C_BOTH]

    xs, ys, errs, ns, pvals = [], [], [], [], []
    for i, (label, grp) in enumerate(groups):
        if len(grp) == 0:
            continue
        rate = grp["align"].mean()
        n_g  = len(grp)
        # Wilson confidence interval
        z = 1.96
        p_hat = rate
        denom = 1 + z**2 / n_g
        center = (p_hat + z**2 / (2*n_g)) / denom
        margin = z * np.sqrt(p_hat*(1-p_hat)/n_g + z**2/(4*n_g**2)) / denom
        xs.append(i); ys.append(rate * 100)
        errs.append(margin * 100); ns.append(n_g)
        binom = stats.binomtest(int(grp["align"].sum()), n_g, p=0.5, alternative="greater")
        pvals.append(binom.pvalue)

    bar_h = ax.barh(xs, ys, color=[bar_colors[i] for i in xs],
                    edgecolor="white", linewidth=1.2, height=0.55)
    ax.errorbar(ys, xs, xerr=errs, fmt="none", color="#333333",
                capsize=4, linewidth=1.5)

    # 50% reference line
    ax.axvline(50, color="#888888", linestyle=":", linewidth=1.5, label="Chance (50%)")

    for i, (rate, n_g, pv) in enumerate(zip(ys, ns, pvals)):
        p_str = f"p={pv:.3f}" if pv >= 0.001 else "p<0.001"
        ax.text(rate + errs[i] + 0.8, i,
                f"{rate:.1f}%  (n={n_g:,}  {p_str})",
                va="center", fontsize=9, color="#333333")

    ax.set_yticks(xs)
    ax.set_yticklabels([lbl for lbl, _ in groups], fontsize=10)
    ax.set_xlabel("% where bill-level leader == RBO source")
    ax.set_title("D.  Alignment Test: Bill-Level Direction\nvs Aggregate RBO Edge Direction")
    ax.set_xlim(0, 135)
    ax.axvline(50, color="#888888", linestyle=":", linewidth=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out = VIZ_DIR / "affiliation_mediation_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/3] Summary panel   -> {out.name}")


# ── Figure 2: Directed network with mediated edges highlighted ────────────────

def fig_network(enr):
    """RBO directed influence network, mediated edges highlighted."""
    # Load GML (has net_influence, out/in_strength node attributes)
    try:
        G = nx.read_gml(str(GML_PATH))
    except Exception:
        # Fall back to building from enriched edge CSV
        G = nx.DiGraph()
        for _, row in enr.iterrows():
            G.add_edge(row["source"], row["target"],
                       weight=float(row["weight"]),
                       net_temporal=int(row["net_temporal"]))
        print("  [network] GML not found; built from enriched CSV (no node attrs)")

    # Identify mediated pairs from enriched CSV
    mediated_edges = set()
    for _, row in enr.dropna(subset=["any_mediation_rate"]).iterrows():
        if row["any_mediation_rate"] > 0:
            mediated_edges.add((row["source"], row["target"]))

    mediated_nodes = set()
    for u, v in mediated_edges:
        mediated_nodes.add(u)
        mediated_nodes.add(v)

    # Select top-K nodes by total involvement, but always include mediated nodes
    involvement = {}
    for node in G.nodes():
        out_s = G.nodes[node].get("out_strength", 0) or sum(d.get("weight", 0)
                for _, _, d in G.out_edges(node, data=True))
        in_s  = G.nodes[node].get("in_strength", 0) or sum(d.get("weight", 0)
                for _, _, d in G.in_edges(node, data=True))
        involvement[node] = out_s + in_s

    TOP_K = 28
    top_nodes = set(sorted(involvement, key=involvement.get, reverse=True)[:TOP_K])
    top_nodes |= mediated_nodes   # ensure mediated nodes always appear
    H = G.subgraph(top_nodes).copy()

    pos = nx.circular_layout(H)

    # Node attributes
    net_inf = {n: H.nodes[n].get("net_influence", 0) for n in H.nodes()}
    inv_arr = np.array([involvement.get(n, 0.1) for n in H.nodes()])
    sizes   = 800 + 3000 * (inv_arr - inv_arr.min()) / (inv_arr.max() - inv_arr.min() + 1e-9)
    node_colors = [
        "#27AE60" if net_inf[n] > 0 else
        "#E74C3C" if net_inf[n] < 0 else
        "#95A5A6"
        for n in H.nodes()
    ]
    # Mediated nodes get a thick gold border
    linewidths   = [4.0 if n in mediated_nodes else 1.0 for n in H.nodes()]
    edge_colors_node = ["#F39C12" if n in mediated_nodes else "#333333" for n in H.nodes()]

    # Edges
    all_edges = list(H.edges(data=True))
    med_edges    = [(u, v, d) for u, v, d in all_edges if (u, v) in mediated_edges]
    unmed_edges  = [(u, v, d) for u, v, d in all_edges if (u, v) not in mediated_edges]

    weights_all  = [d.get("weight", 0.01) for _, _, d in all_edges]
    w_min, w_max = min(weights_all), max(weights_all)

    def edge_width(w):
        return 0.5 + 4.0 * (w - w_min) / (w_max - w_min + 1e-9)

    fig, ax = plt.subplots(figsize=(18, 18))

    # Draw unmediated edges first (behind)
    if unmed_edges:
        nx.draw_networkx_edges(
            H, pos,
            edgelist=[(u, v) for u, v, _ in unmed_edges],
            width=[edge_width(d.get("weight", 0.01)) for _, _, d in unmed_edges],
            alpha=0.25, edge_color=C_UNMED,
            arrows=True, arrowsize=10,
            connectionstyle="arc3,rad=0.12", ax=ax,
        )

    # Draw mediated edges (front, bold)
    if med_edges:
        nx.draw_networkx_edges(
            H, pos,
            edgelist=[(u, v) for u, v, _ in med_edges],
            width=[max(3.5, edge_width(d.get("weight", 0.01)) * 2) for _, _, d in med_edges],
            alpha=0.95, edge_color=AMBER,
            arrows=True, arrowsize=25,
            connectionstyle="arc3,rad=0.18", ax=ax,
        )

    # Nodes
    nx.draw_networkx_nodes(
        H, pos, node_size=sizes, node_color=node_colors,
        edgecolors=edge_colors_node, linewidths=linewidths, ax=ax,
    )

    # Mediated node labels (larger, bold)
    med_label_dict   = {n: n.title() for n in H.nodes() if n in mediated_nodes}
    other_label_dict = {n: n.title() for n in H.nodes() if n not in mediated_nodes}

    nx.draw_networkx_labels(H, pos, labels=other_label_dict,
                            font_size=8, font_weight="normal",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6), ax=ax)
    nx.draw_networkx_labels(H, pos, labels=med_label_dict,
                            font_size=10, font_weight="bold",
                            bbox=dict(facecolor="#FEF9E7", edgecolor="#F39C12",
                                      alpha=0.95, linewidth=1.5), ax=ax)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="#27AE60",  label="Net influencer (net_influence > 0)"),
        mpatches.Patch(facecolor="#E74C3C",  label="Net follower (net_influence < 0)"),
        mpatches.Patch(facecolor="#95A5A6",  label="Neutral (net_influence = 0)"),
        mpatches.Patch(facecolor=AMBER,      label="Affiliation-mediated edge",  edgecolor="white"),
        mpatches.Patch(facecolor="white",    label="Mediated node (gold border)", edgecolor="#F39C12",
                       linewidth=2.5),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=11,
              framealpha=0.92, edgecolor="#DDDDDD")

    ax.set_title(
        f"RBO Directed Influence Network — Top {len(H)} Nodes\n"
        "Affiliation-mediated edges highlighted (amber); mediated nodes in gold borders",
        fontsize=15, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    out = VIZ_DIR / "affiliation_mediation_network.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/3] Network figure  -> {out.name}")


# ── Figure 3: Mediated adoption timeline ─────────────────────────────────────

def fig_timeline(med):
    """Arrow timeline for each affiliation-mediated bill-level directed adoption."""
    med_dir = med[med["is_bill_directed"] & med["is_any_mediated"]].copy()

    # Build display label: "Source → Target\n(bill)"
    med_dir["pair_label"] = (
        med_dir["rbo_source"].str.title() + " \u2192 "
        + med_dir["rbo_target"].str.title()
        + "\n(" + med_dir["bill"] + ")"
    )
    # Sort by source then bill
    med_dir = med_dir.sort_values(["rbo_source", "bill"]).reset_index(drop=True)

    # Channel label for annotation
    def channel(row):
        if row["is_lobbyist_mediated"] and row["is_firm_mediated"]:
            return "Lobbyist + Firm"
        if row["is_lobbyist_mediated"]:
            return "Lobbyist"
        return "Firm"

    med_dir["channel"] = med_dir.apply(channel, axis=1)
    ch_colors = {"Lobbyist": C_LOB_ONLY, "Firm": C_FIRM_ONLY, "Lobbyist + Firm": C_BOTH}

    fig, ax = plt.subplots(figsize=(13, max(5, len(med_dir) * 1.3)))

    qs = range(1, 9)
    ax.set_xticks(list(qs))
    ax.set_xticklabels([QUARTER_LABELS[q] for q in qs], fontsize=10, rotation=25, ha="right")

    # Vertical quarter grid
    for q in qs:
        ax.axvline(q, color="#E0E0E0", linewidth=0.8, zorder=0)

    for i, (_, row) in enumerate(med_dir.iterrows()):
        col = ch_colors[row["channel"]]
        ql, qf = row["q_leader"], row["q_follower"]

        # Leader dot
        ax.scatter(ql, i, s=150, color=EMERALD, zorder=5, edgecolors="white", linewidths=1.5)
        # Follower dot
        ax.scatter(qf, i, s=150, color=CRIMSON,  zorder=5, edgecolors="white", linewidths=1.5)
        # Arrow from leader → follower
        ax.annotate(
            "", xy=(qf, i), xytext=(ql, i),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                            mutation_scale=18, shrinkA=8, shrinkB=8),
        )
        # Lag label above arrow
        ax.text((ql + qf) / 2, i + 0.28,
                f"lag = {int(row['lag_quarters'])} q  ·  {row['channel']}",
                ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")

        # Shared intermediary names (below arrow, smaller)
        if pd.notna(row["shared_lobbyists"]) and row["shared_lobbyists"]:
            lobs = str(row["shared_lobbyists"]).split("|")
            preview = ", ".join(l.split(",")[0] for l in lobs[:3])
            if len(lobs) > 3:
                preview += f" +{len(lobs)-3} more"
            ax.text((ql + qf) / 2, i - 0.32,
                    f"Shared lobbyist(s): {preview}",
                    ha="center", va="top", fontsize=7.5, color="#555555", style="italic")
        if pd.notna(row["shared_firms"]) and row["shared_firms"]:
            ax.text((ql + qf) / 2, i - 0.54,
                    f"Shared firm(s): {row['shared_firms']}",
                    ha="center", va="top", fontsize=7.5, color="#555555", style="italic")

    ax.set_yticks(range(len(med_dir)))
    ax.set_yticklabels(med_dir["pair_label"], fontsize=10)
    ax.set_xlim(0.3, 8.7)
    ax.set_ylim(-0.9, len(med_dir) - 0.2)
    ax.set_xlabel("Congress quarter", fontsize=11)
    ax.set_title(
        "Affiliation-Mediated Bill Adoptions — Adoption Timeline\n"
        "Green dot = leader (first adopter)  ·  Red dot = follower  ·  "
        "Arrow color = transmission channel",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=EMERALD,  label="Bill-level leader (first adopter)"),
        mpatches.Patch(color=CRIMSON,  label="Follower (later adopter)"),
        mpatches.Patch(color=C_LOB_ONLY,  label="Lobbyist channel"),
        mpatches.Patch(color=C_FIRM_ONLY, label="Firm channel"),
        mpatches.Patch(color=C_BOTH,      label="Both channels"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = VIZ_DIR / "affiliation_mediation_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/3] Timeline figure -> {out.name}")


# -- Main ---------------------------------------------------------------------

def main():
    print("Loading data...")
    med, enr = load_data()
    print(f"  Loaded {len(med):,} (edge, bill) records  |  {len(enr):,} RBO edges")

    directed = med[med["is_bill_directed"]]
    print(f"  Directed pairs: {len(directed):,}  |  "
          f"Bill-level mediated: {directed['is_any_mediated'].sum()}  |  "
          f"Network-connected: {directed['net_any_connected'].sum()}")

    print("\nGenerating figures...")
    fig_summary(med)
    fig_network(enr)
    fig_timeline(med)
    print(f"\nAll figures saved to {VIZ_DIR}")


if __name__ == "__main__":
    main()
