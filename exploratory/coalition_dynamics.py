"""
Coalition dynamics charts: per-quarter new entrants and cumulative unique
client count for the top 2 bills per sector (6 sectors).

Layout: 6 rows (sectors) × 2 cols (bills). Each subplot shows:
  - Positive bars: new entrants (first-time clients) per quarter
  - Line + markers: cumulative unique client count (right axis)
  - Dashed vertical line: quarter after key legislative event
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.formatter.useoffset": False})

# ── Bill metadata ─────────────────────────────────────────────────────────────
# (congress, bill_number, sector, short_title, code_label,
#  fate_label, fate_color, line_idx)
BILLS = [
    # Technology
    (116, "H.R.1044",
     "Technology", "Fairness for High-Skilled\nImmigrants Act",
     "TEC/IMM · 116th",
     "Passed House Jul '19\n(Died in Senate)", "#DC2626", 3),

    (116, "H.R.1644",
     "Technology", "Save the Internet Act\n(Net Neutrality)",
     "TEC · 116th",
     "Passed House Apr '19\n(Died in Senate)", "#DC2626", 2),

    # Healthcare
    (115, "H.R.1628",
     "Healthcare", "American Health Care Act\n(ACA Repeal Attempt)",
     "HCR · 115th",
     "Failed Senate Jul '17", "#DC2626", 3),

    (116, "S.1895",
     "Healthcare", "Lower Health Care\nCosts Act",
     "HCR · 116th",
     "Died in committee\n(end of 116th Cong.)", "#DC2626", None),

    # Energy
    (115, "S.1460",
     "Energy", "Energy & Natural\nResources Act of 2017",
     "ENG · 115th",
     "Died in committee\n(end of 115th Cong.)", "#DC2626", None),

    (116, "H.R.360",
     "Energy", "Cyber Sense Act\n(Grid Cybersecurity)",
     "ENG · 116th",
     "Enacted Dec '20\n(Div. Y, CAA 2021)", "#16A34A", None),

    # Defense
    (115, "H.R.2810",
     "Defense", "NDAA FY2018",
     "DEF · 115th",
     "Enacted Dec 12 '17", "#16A34A", 4),

    (116, "H.R.2500",
     "Defense", "NDAA FY2020",
     "DEF · 116th",
     "Enacted Dec 20 '19", "#16A34A", 4),

    # Finance / Banking
    (115, "H.R.10",
     "Finance/Banking", "Financial CHOICE Act\n(Dodd-Frank Reform)",
     "FIN · 115th",
     "Passed House Jun '17\n(Died in Senate)", "#DC2626", 2),

    (116, "H.R.1994",
     "Finance/Banking", "SECURE Act\n(Retirement Reform)",
     "FIN · 116th",
     "Enacted Dec 20 '19", "#16A34A", 4),

    # Transportation
    (115, "S.1405",
     "Transportation", "FAA Reauthorization\nAct of 2017",
     "TRA · 115th",
     "Superseded by H.R.302\n(enacted Oct '18)", "#D97706", 7),

    (116, "S.2302",
     "Transportation", "America's Transportation\nInfrastructure Act of 2019",
     "TRA · 116th",
     "Died in Senate\n(end of 116th Cong.)", "#DC2626", None),
]

QUARTERS = {
    115: ["2017_q1","2017_q2","2017_q3","2017_q4",
          "2018_q1","2018_q2","2018_q3","2018_q4"],
    116: ["2019_q1","2019_q2","2019_q3","2019_q4",
          "2020_q1","2020_q2","2020_q3","2020_q4"],
}
QLABELS = {
    115: ["Q1'17","Q2'17","Q3'17","Q4'17","Q1'18","Q2'18","Q3'18","Q4'18"],
    116: ["Q1'19","Q2'19","Q3'19","Q4'19","Q1'20","Q2'20","Q3'20","Q4'20"],
}
SECTOR_COLORS = {
    "Technology":      "#2563EB",
    "Healthcare":      "#16A34A",
    "Energy":          "#D97706",
    "Defense":         "#DC2626",
    "Finance/Banking": "#7C3AED",
    "Transportation":  "#0891B2",
}

# ── Load ──────────────────────────────────────────────────────────────────────
def load(congress):
    base = (f"/sessions/serene-lucid-franklin/mnt/Independent Study/"
            f"Lobbying-Networks-Research/data/congress/{congress}")
    r = pd.read_csv(f"{base}/opensecrets_lda_reports.csv")
    r["quarter_base"] = r["report_type"].str.extract(r"^(q\d)")
    r["quarter_key"]  = r["year"].astype(str) + "_" + r["quarter_base"]
    return r

raw = {115: load(115), 116: load(116)}

def client_sets_per_quarter(congress, bill):
    """Return dict {quarter_key: set_of_clients}."""
    df = raw[congress]
    sub = (df[df["bill_number"] == bill]
             .groupby(["quarter_key","client"])
             .size().reset_index()[["quarter_key","client"]])
    qkeys = QUARTERS[congress]
    return {q: set(sub[sub["quarter_key"] == q]["client"]) for q in qkeys}

def coalition_dynamics(congress, bill):
    """Returns arrays (length 8) for: new_entrants, cumulative_unique."""
    csets = client_sets_per_quarter(congress, bill)
    qkeys = QUARTERS[congress]
    seen_ever = set()
    new_arr, cum_arr = [], []

    for q in qkeys:
        curr = csets[q]
        new_e = curr - seen_ever
        seen_ever |= curr
        new_arr.append(len(new_e))
        cum_arr.append(len(seen_ever))

    return np.array(new_arr), np.array(cum_arr)

# ── Figure ────────────────────────────────────────────────────────────────────
sectors = ["Technology","Healthcare","Energy","Defense","Finance/Banking","Transportation"]
fig, axes = plt.subplots(6, 2, figsize=(14, 26))
fig.patch.set_facecolor("#F8FAFC")

for row, sector in enumerate(sectors):
    sector_bills = [b for b in BILLS if b[2] == sector]

    for col, (congress, bill, sec, title, code_label,
               fate_label, fate_color, line_idx) in enumerate(sector_bills):
        ax = axes[row, col]
        ax.set_facecolor("#FFFFFF")
        color = SECTOR_COLORS[sector]

        new_e, cum = coalition_dynamics(congress, bill)
        xs = np.arange(8)
        max_cum = max(cum.max(), 1)
        y_max = max(new_e.max(), 1)

        # ── Left axis: new entrant bars ───────────────────────────────────
        bar_w = 0.55
        ax.bar(xs, new_e, width=bar_w, color=color, alpha=0.75,
               label="New entrants", zorder=3)

        ax.set_ylim(0, y_max * 3.0)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
        ax.set_ylabel("New entrants", fontsize=8, color="#64748B")
        ax.tick_params(axis="y", labelsize=8)

        # ── Right axis: cumulative unique clients ─────────────────────────
        ax2 = ax.twinx()
        ax2.plot(xs, cum, color=color, linewidth=2.2,
                 marker="o", markersize=5.5,
                 markerfacecolor="white", markeredgewidth=2.0,
                 markeredgecolor=color, zorder=5, linestyle="-")
        ax2.set_ylim(0, max_cum * 2.6)
        ax2.set_ylabel("Cumul. unique clients", fontsize=8, color=color)
        ax2.tick_params(axis="y", labelcolor=color, labelsize=8)
        ax2.spines["right"].set_color(color)
        ax2.spines["right"].set_alpha(0.4)

        # Final cumulative annotation
        ax2.annotate(f"{cum[-1]} total",
                     xy=(7, cum[-1]),
                     xytext=(4, 6), textcoords="offset points",
                     fontsize=8.5, fontweight="bold", color=color, ha="left")

        # ── Fate line ─────────────────────────────────────────────────────
        if line_idx is not None and 0 <= line_idx <= 7:
            lx = line_idx - 0.5
            ax.axvline(lx, color=fate_color, linewidth=1.6,
                       linestyle="--", alpha=0.85, zorder=6)
            ax.text(lx + 0.1, y_max * 2.7, fate_label,
                    fontsize=7, color=fate_color, va="top", ha="left",
                    linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=fate_color, alpha=0.88, lw=0.8))
        else:
            ax.text(0.98, 0.97, fate_label,
                    transform=ax.transAxes,
                    fontsize=6.8, color=fate_color, va="top", ha="right",
                    linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=fate_color, alpha=0.88, lw=0.8))

        # ── Axes formatting ───────────────────────────────────────────────
        ax.set_xticks(xs)
        ax.set_xticklabels(QLABELS[congress], fontsize=8.5,
                           rotation=30, ha="right")
        ax.set_xlim(-0.6, 8.3)
        ax.spines[["top"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#CBD5E1")
        ax2.spines[["top"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3,
                color="#E2E8F0", zorder=0)

        ax.set_xlabel(f"{bill}  ·  {code_label}",
                      fontsize=8, color="#94A3B8", labelpad=4)
        ax.set_title(title, fontsize=10.5, fontweight="bold",
                     color="#1E293B", pad=6, loc="left")

    # Sector label
    axes[row, 0].text(-0.22, 0.5, sector,
                      transform=axes[row, 0].transAxes,
                      rotation=90, ha="center", va="center",
                      fontsize=11.5, fontweight="bold",
                      color=SECTOR_COLORS[sector])

# Column headers
fig.text(0.305, 0.991, "Bill 1 (higher client count)",
         ha="center", va="top", fontsize=12, fontweight="bold", color="#334155")
fig.text(0.735, 0.991, "Bill 2 (lower client count)",
         ha="center", va="top", fontsize=12, fontweight="bold", color="#334155")

# Legend
e_patch   = mpatches.Patch(color="#64748B", alpha=0.75,
                            label="New entrants per quarter (bars)")
cum_patch = plt.Line2D([0],[0], color="#64748B", linewidth=2,
                       marker="o", markerfacecolor="white",
                       markeredgecolor="#64748B", label="Cumul. unique clients (right axis)")
g_line    = plt.Line2D([0],[0], color="#16A34A", linestyle="--",
                       linewidth=1.5, label="Enacted/resolved")
r_line    = plt.Line2D([0],[0], color="#DC2626", linestyle="--",
                       linewidth=1.5, label="Failed/died")
o_line    = plt.Line2D([0],[0], color="#D97706", linestyle="--",
                       linewidth=1.5, label="Superseded")

fig.legend(handles=[e_patch, cum_patch, g_line, r_line, o_line],
           loc="upper center", bbox_to_anchor=(0.52, 0.988),
           ncol=3, fontsize=8.5, frameon=True,
           edgecolor="#CBD5E1", facecolor="white")

fig.text(0.52, 1.003,
         "Lobbying Coalition Dynamics — New Entrants & Cumulative Clients",
         ha="center", va="top", fontsize=15.5,
         fontweight="bold", color="#0F172A")
fig.text(0.52, 0.995,
         "Fortune-linked clients  ·  per-quarter first-time entrants (bars)  ·  dashed = quarter after key legislative event",
         ha="center", va="top", fontsize=9, color="#64748B")

plt.subplots_adjust(left=0.12, right=0.88, top=0.952, bottom=0.03,
                    hspace=0.85, wspace=0.52)

out = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
       "Lobbying-Networks-Research/visualizations/coalition_dynamics.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.close()
