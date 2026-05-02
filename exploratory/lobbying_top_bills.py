"""
Single-sector 'champion' chart: the top-lobbied bill in each sector
across both the 115th and 116th Congresses combined.
6 subplots (2x3), one per sector. Vertical fate line included.
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

# ── Top bill per sector ───────────────────────────────────────────────────────
# (congress, bill_number, sector_label, short_title, issue_code,
#  fate_label, fate_color, line_idx)
# line_idx: 0-7 within congress's 8-quarter window; None = outside window
TOP_BILLS = [
    (116, "H.R.1644",
     "Technology",
     "Save the Internet Act\n(Net Neutrality)",
     "TEC  ·  $23.1M total allocated",
     "Passed House Apr '19\n(Died in Senate)", "#DC2626", 2),

    (115, "H.R.1628",
     "Healthcare",
     "American Health Care Act\n(ACA Repeal Attempt)",
     "HCR  ·  $74.1M total allocated",
     "Failed Senate Jul '17", "#DC2626", 3),

    (115, "S.1460",
     "Energy / Oil & Gas",
     "Energy & Natural Resources\nAct of 2017",
     "ENG  ·  $48.5M total allocated",
     "Died in committee\n(end of 115th Cong.)", "#DC2626", None),

    (115, "H.R.2810",
     "Defense",
     "National Defense\nAuthorization Act FY2018",
     "DEF  ·  $29.3M total allocated",
     "Enacted Dec 12 '17", "#16A34A", 4),

    (115, "H.R.10",
     "Finance / Banking",
     "Financial CHOICE Act\n(Dodd-Frank Reform)",
     "FIN  ·  $21.1M total allocated",
     "Passed House Jun '17\n(Died in Senate)", "#DC2626", 2),

    (116, "H.R.1044",
     "Tech Workforce",
     "Fairness for High-Skilled\nImmigrants Act",
     "TEC/IMM  ·  $164.3M total allocated",
     "Passed House Jul '19\n(Died in Senate)", "#DC2626", 3),
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
COLORS = {
    "Technology":      "#2563EB",
    "Healthcare":      "#16A34A",
    "Energy / Oil & Gas": "#D97706",
    "Defense":         "#DC2626",
    "Finance / Banking": "#7C3AED",
    "Tech Workforce":  "#0891B2",
}

def load(congress):
    base = (f"/sessions/serene-lucid-franklin/mnt/Independent Study/"
            f"Lobbying-Networks-Research/data/congress/{congress}")
    r = pd.read_csv(f"{base}/opensecrets_lda_reports.csv")
    r["quarter_base"] = r["report_type"].str.extract(r"^(q\d)")
    r["quarter_key"]  = r["year"].astype(str) + "_" + r["quarter_base"]
    return r

raw = {115: load(115), 116: load(116)}

def quarterly_spend(congress, bill):
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

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#F8FAFC")
axes_flat = axes.flatten()

for ax, (congress, bill, sector, title, code_label,
          fate_label, fate_color, line_idx) in zip(axes_flat, TOP_BILLS):

    ax.set_facecolor("#FFFFFF")
    color = COLORS[sector]

    qtotals = quarterly_spend(congress, bill)
    cumvals = np.cumsum(qtotals) / 1e6
    qbars   = qtotals / 1e6
    xs      = np.arange(8)
    ymax    = max(cumvals[-1] * 1.38, 0.5)

    # Bars + cumulative line
    ax.bar(xs, qbars, color=color, alpha=0.20, width=0.65, zorder=2)
    ax.fill_between(xs, cumvals, alpha=0.10, color=color, zorder=1)
    ax.plot(xs, cumvals, color=color, linewidth=2.6,
            marker="o", markersize=6, markerfacecolor="white",
            markeredgewidth=2.2, markeredgecolor=color, zorder=4)

    # Final value
    ax.annotate(f"${cumvals[-1]:.1f}M",
                xy=(7, cumvals[-1]),
                xytext=(-6, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color=color, ha="right")

    # Fate vertical line
    if line_idx is not None:
        lx = line_idx - 0.5
        ax.axvline(lx, color=fate_color, linewidth=1.8,
                   linestyle="--", alpha=0.9, zorder=5)
        ax.text(lx + 0.12, ymax * 0.97, fate_label,
                fontsize=8, color=fate_color, va="top", ha="left",
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=fate_color, alpha=0.9, lw=0.9))
    else:
        ax.text(7.35, ymax * 0.97, fate_label,
                fontsize=7.5, color=fate_color, va="top", ha="right",
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=fate_color, alpha=0.9, lw=0.9))

    ax.set_xticks(xs)
    ax.set_xticklabels(QLABELS[congress], fontsize=9, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=False))
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(-0.6, 8.3)
    ax.set_ylim(0, ymax)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#CBD5E1")
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#E2E8F0", zorder=0)

    # Sector chip
    ax.text(-0.02, 1.07, f"  {sector}  ",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"))

    ax.set_title(f"{bill}  ·  {congress}th Congress\n{code_label}",
                 fontsize=8.5, color="#64748B", pad=4, loc="right")
    ax.text(-0.02, 1.0, title,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#1E293B", va="top")

# Legend
enacted_patch = mpatches.Patch(color="#16A34A", label="Enacted / signed into law")
failed_patch  = mpatches.Patch(color="#DC2626",
                               label="Failed / died (or passed one chamber only)")
fig.legend(handles=[enacted_patch, failed_patch],
           loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=10, frameon=True,
           edgecolor="#CBD5E1", facecolor="white")

fig.text(0.5, 1.02,
         "Top-Lobbied Bill per Sector — 115th & 116th Congress",
         ha="center", va="top", fontsize=17, fontweight="bold", color="#0F172A")
fig.text(0.5, 1.006,
         "Fortune-linked clients  ·  amount_allocated (cumulative)  ·  dashed line = quarter after key legislative event",
         ha="center", va="top", fontsize=9.5, color="#64748B")

plt.tight_layout(rect=[0, 0.04, 1, 1.0])
plt.subplots_adjust(hspace=0.62, wspace=0.32)

out = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
       "Lobbying-Networks-Research/visualizations/lobbying_top_bills_by_sector.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.close()
