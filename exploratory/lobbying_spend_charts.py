"""
Cumulative lobbying spend charts for 12 sector-specific bills,
115th (2017-2018) vs 116th (2019-2020) Congress.
Vertical dashed line marks the quarter AFTER the bill's key legislative event.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.formatter.useoffset": False,
})

# ── Bill metadata ─────────────────────────────────────────────────────────────
# (congress, bill_number, short_title, sector, issue_code,
#  fate_label, fate_color, line_idx)
#  line_idx = 0-7 for quarter to draw line at; None = outside window (annotate only)
BILLS = [
    (115, "S.19",
     "MOBILE NOW Act\n(Spectrum Infrastructure)",
     "Technology", "TEC",
     "Enacted Mar '18", "#16A34A", 5),          # signed Q1'18 → line at Q2'18

    (116, "H.R.1644",
     "Save the Internet Act\n(Net Neutrality)",
     "Technology", "TEC",
     "Passed House Apr '19\n(Died in Senate)", "#DC2626", 2),   # passed House Q2'19 → line at Q3'19

    (115, "H.R.1628",
     "American Health Care Act\n(ACA Repeal Attempt)",
     "Healthcare", "HCR",
     "Failed Senate Jul '17", "#DC2626", 3),    # failed Q3'17 → line at Q4'17

    (116, "S.1895",
     "Lower Health Care\nCosts Act",
     "Healthcare", "HCR",
     "Died in committee\n(end of 116th Cong.)", "#DC2626", None),

    (115, "S.1460",
     "Energy & Natural\nResources Act of 2017",
     "Energy", "ENG",
     "Died in committee\n(end of 115th Cong.)", "#DC2626", None),

    (116, "S.2602",
     "American Energy\nInnovation Act",
     "Energy", "ENG",
     "Cloture failed Mar '20", "#DC2626", 5),   # failed Q1'20 → line at Q2'20

    (115, "H.R.10",
     "Financial CHOICE Act\n(Dodd-Frank Reform)",
     "Finance", "FIN",
     "Passed House Jun '17\n(Died in Senate)", "#DC2626", 2),   # passed House Q2'17 → line at Q3'17

    (116, "H.R.1994",
     "SECURE Act\n(Retirement Reform)",
     "Finance", "FIN",
     "Enacted Dec '19", "#16A34A", 4),          # signed Q4'19 → line at Q1'20

    (115, "H.R.2810",
     "NDAA FY2018",
     "Defense", "DEF",
     "Enacted Dec '17", "#16A34A", 4),          # signed Q4'17 → line at Q1'18

    (116, "H.R.2500",
     "NDAA FY2020",
     "Defense", "DEF",
     "Enacted Dec '19", "#16A34A", 4),          # signed Q4'19 → line at Q1'20

    (115, "S.2155",
     "Econ. Growth, Reg. Relief\n& Consumer Prot. Act",
     "Banking", "BAN",
     "Enacted May '18", "#16A34A", 6),          # signed Q2'18 → line at Q3'18

    (116, "H.R.1595",
     "SAFE Banking Act\n(Cannabis Banking)",
     "Banking", "BAN",
     "Passed House Sep '19\n(Died in Senate)", "#DC2626", 3),   # passed House Q3'19 → line at Q4'19
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
    "Technology": "#2563EB",
    "Healthcare":  "#16A34A",
    "Energy":      "#D97706",
    "Finance":     "#7C3AED",
    "Defense":     "#DC2626",
    "Banking":     "#0891B2",
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load(congress):
    base = (f"/sessions/serene-lucid-franklin/mnt/Independent Study/"
            f"Lobbying-Networks-Research/data/congress/{congress}")
    r = pd.read_csv(f"{base}/opensecrets_lda_reports.csv")
    r["quarter_base"] = r["report_type"].str.extract(r"^(q\d)")
    r["quarter_key"]  = r["year"].astype(str) + "_" + r["quarter_base"]
    return r

raw = {115: load(115), 116: load(116)}

def quarterly_spend(congress, bill):
    """Per-quarter allocated spend, deduplicated at filing × quarter level."""
    df = raw[congress]
    sub = df[df["bill_number"] == bill][["uniq_id","quarter_key","amount_allocated"]].copy()
    sub = sub.groupby(["uniq_id","quarter_key"], as_index=False)["amount_allocated"].max()
    q_totals = sub.groupby("quarter_key")["amount_allocated"].sum()
    return q_totals.reindex(QUARTERS[congress], fill_value=0).values

def fmt_millions(x, _):
    if x >= 100: return f"${x:.0f}M"
    if x >= 10:  return f"${x:.0f}M"
    if x >= 1:   return f"${x:.1f}M"
    return f"${x:.2f}M"

# ── Build figure ──────────────────────────────────────────────────────────────
sectors = ["Technology","Healthcare","Energy","Finance","Defense","Banking"]
fig, axes = plt.subplots(6, 2, figsize=(14, 25))
fig.patch.set_facecolor("#F8FAFC")

for row, sector in enumerate(sectors):
    sector_bills = [b for b in BILLS if b[3] == sector]

    for col, (congress, bill, title, sec, code,
              fate_label, fate_color, line_idx) in enumerate(sector_bills):
        ax = axes[row, col]
        ax.set_facecolor("#FFFFFF")
        color = SECTOR_COLORS[sector]

        qtotals = quarterly_spend(congress, bill)
        cumvals = np.cumsum(qtotals) / 1e6
        qbars   = qtotals / 1e6
        xs      = np.arange(8)
        ymax    = max(cumvals[-1] * 1.35, 0.1)

        # Per-quarter bars (faint background)
        ax.bar(xs, qbars, color=color, alpha=0.18, width=0.65, zorder=2)

        # Cumulative line
        ax.fill_between(xs, cumvals, alpha=0.10, color=color, zorder=1)
        ax.plot(xs, cumvals, color=color, linewidth=2.4,
                marker="o", markersize=5.5,
                markerfacecolor="white", markeredgewidth=2.0,
                markeredgecolor=color, zorder=4)

        # Final value annotation
        ax.annotate(f"${cumvals[-1]:.1f}M",
                    xy=(7, cumvals[-1]),
                    xytext=(-6, 9), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color, ha="right")

        # ── Vertical fate line ────────────────────────────────────────────────
        if line_idx is not None and 0 <= line_idx <= 7:
            # Line drawn between (line_idx - 0.5) and (line_idx + 0.5)
            # but cleaner to draw at x = line_idx - 0.5 (boundary between quarters)
            lx = line_idx - 0.5
            ax.axvline(lx, color=fate_color, linewidth=1.6,
                       linestyle="--", alpha=0.85, zorder=5)
            # Fate annotation at top of line
            ax.text(lx + 0.12, ymax * 0.97, fate_label,
                    fontsize=7, color=fate_color, va="top", ha="left",
                    linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=fate_color, alpha=0.85, lw=0.8))
        else:
            # Outside window: annotate at right edge
            ax.text(7.4, ymax * 0.97, fate_label,
                    fontsize=6.8, color=fate_color, va="top", ha="right",
                    linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=fate_color, alpha=0.85, lw=0.8))

        # Axes
        ax.set_xticks(xs)
        ax.set_xticklabels(QLABELS[congress], fontsize=8.5, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=False))
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlim(-0.6, 8.2)
        ax.set_ylim(0, ymax)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#CBD5E1")
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="#E2E8F0", zorder=0)

        ax.set_xlabel(f"{bill}  ·  {congress}th Cong.  ·  {code}",
                      fontsize=8, color="#94A3B8", labelpad=4)
        ax.set_title(title, fontsize=10.5, fontweight="bold",
                     color="#1E293B", pad=6, loc="left")

    # Sector label
    axes[row, 0].text(-0.20, 0.5, sector,
                      transform=axes[row, 0].transAxes,
                      rotation=90, ha="center", va="center",
                      fontsize=12, fontweight="bold",
                      color=SECTOR_COLORS[sector])

# Column headers
for col, label in enumerate(["115th Congress  (2017–2018)", "116th Congress  (2019–2020)"]):
    fig.text(0.30 + col * 0.46, 0.991, label,
             ha="center", va="top", fontsize=13.5,
             fontweight="bold", color="#334155")

# Legend
enacted_patch = mpatches.Patch(color="#16A34A", label="Enacted / signed into law")
failed_patch  = mpatches.Patch(color="#DC2626", label="Failed / died / passed one chamber only")
fig.legend(handles=[enacted_patch, failed_patch],
           loc="upper center", bbox_to_anchor=(0.52, 0.987),
           ncol=2, fontsize=9, frameon=True,
           edgecolor="#CBD5E1", facecolor="white")

# Titles
fig.text(0.52, 1.003,
         "Cumulative Lobbying Spend — Sector-Specific Bills",
         ha="center", va="top", fontsize=16,
         fontweight="bold", color="#0F172A")
fig.text(0.52, 0.995,
         "Fortune-linked clients  |  amount_allocated per quarter  |  dashed line = quarter after key legislative event",
         ha="center", va="top", fontsize=9, color="#64748B")

plt.subplots_adjust(left=0.11, right=0.97, top=0.953, bottom=0.03,
                    hspace=0.82, wspace=0.36)

out = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
       "Lobbying-Networks-Research/visualizations/lobbying_spend_by_sector.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.close()
