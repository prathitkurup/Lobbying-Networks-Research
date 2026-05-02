"""
Bill agenda carryover — 115th Congress.

Tests whether the aggregate bill agenda set in Y1 (2017, Q1–Q4) persists into
Y2 (2018, Q5–Q8). For each Y2 quarter: what fraction of lobbied bills were
already present in any Y1 quarter? Also reports cumulative Y1 bill introduction
pace to show how quickly the agenda crystallises.

Outputs (outputs/analysis/):
  09_agenda_carryover.png
  09_agenda_carryover.txt
"""

import sys, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ROOT

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESS   = 115
DATA_PATH  = ROOT / "data" / "congress" / str(CONGRESS) / "opensecrets_lda_reports.csv"
OUT_DIR    = ROOT / "outputs" / "analysis"

# Y1 = first congressional year; Y2 = second
Y1_YEAR    = 2017
Y2_YEAR    = 2018

# Quarter order within each year (base + amended variants all included)
Y1_QUARTER_ORDER = ["q1", "q2", "q3", "q4"]
Y2_QUARTER_ORDER = ["q1", "q2", "q3", "q4"]   # year 2018 still files as q1..q4

# Plot aesthetics
C_CARRYOVER = "#4C72B0"
C_NEW       = "#DD8452"
C_LINE      = "#2ca02c"

# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load 115th Congress LDA reports; drop rows without a bill number."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["bill_number"])
    df["bill_number"] = df["bill_number"].str.strip().str.upper()
    # Normalise report_type to base quarter (strip amendment suffixes a/t/ta)
    df["base_quarter"] = (
        df["report_type"]
        .str.extract(r"^(q[1-4])", expand=False)
    )
    return df

# ---------------------------------------------------------------------------
# Aggregate bill sets
# ---------------------------------------------------------------------------

def get_bills_by_quarter(df: pd.DataFrame, year: int) -> dict[str, set]:
    """Return {base_quarter -> set of unique bill numbers} for a given year."""
    sub = df[df["year"] == year]
    quarters = {}
    for q in ["q1", "q2", "q3", "q4"]:
        bills = sub[sub["base_quarter"] == q]["bill_number"].unique()
        quarters[q] = set(bills)
    return quarters

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame) -> dict:
    """Compute carryover statistics between Y1 and Y2."""
    y1_quarters = get_bills_by_quarter(df, Y1_YEAR)
    y2_quarters = get_bills_by_quarter(df, Y2_YEAR)

    # Full Y1 bill universe (all four quarters combined)
    y1_all = set().union(*y1_quarters.values())

    # Cumulative Y1 bill set as quarters accumulate
    y1_cumulative = {}
    seen = set()
    for q in Y1_QUARTER_ORDER:
        seen = seen | y1_quarters[q]
        y1_cumulative[q] = set(seen)

    # Per Y2 quarter: carryover = bills already in y1_all; new = not in y1_all
    y2_rows = []
    for q in Y2_QUARTER_ORDER:
        bills = y2_quarters[q]
        carryover = bills & y1_all
        new       = bills - y1_all
        y2_rows.append({
            "quarter":         f"2018-{q.upper()}",
            "total":           len(bills),
            "carryover":       len(carryover),
            "new":             len(new),
            "carryover_frac":  len(carryover) / len(bills) if bills else 0,
        })

    # Overall Y2 vs Y1 overlap
    y2_all = set().union(*y2_quarters.values())
    overall_carryover = y2_all & y1_all
    overall_new       = y2_all - y1_all
    y1_not_in_y2      = y1_all - y2_all

    return {
        "y1_all":          y1_all,
        "y2_all":          y2_all,
        "y1_cumulative":   y1_cumulative,
        "y2_rows":         y2_rows,
        "overall_carryover": overall_carryover,
        "overall_new":       overall_new,
        "y1_not_in_y2":      y1_not_in_y2,
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(results: dict, out_path: Path) -> None:
    """Three-panel figure: stacked bar, cumulative Y1 growth, overall overlap."""
    y2_rows      = results["y2_rows"]
    y1_cumul     = results["y1_cumulative"]
    y1_all       = results["y1_all"]
    y2_all       = results["y2_all"]
    n_carryover  = len(results["overall_carryover"])
    n_new        = len(results["overall_new"])
    n_y1_only    = len(results["y1_not_in_y2"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "115th Congress — Bill Agenda Carryover from Y1 (2017) to Y2 (2018)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ------------------------------------------------------------------
    # Panel 1: Stacked bar — carryover vs. new per Y2 quarter
    # ------------------------------------------------------------------
    ax = axes[0]
    quarters  = [r["quarter"] for r in y2_rows]
    carryover = [r["carryover"] for r in y2_rows]
    new_bills = [r["new"] for r in y2_rows]
    fracs     = [r["carryover_frac"] for r in y2_rows]
    x         = range(len(quarters))

    bars_c = ax.bar(x, carryover, color=C_CARRYOVER, label="Already in Y1")
    bars_n = ax.bar(x, new_bills, bottom=carryover, color=C_NEW, label="New in Y2")

    for i, (c, n, f) in enumerate(zip(carryover, new_bills, fracs)):
        ax.text(i, c + n + 5, f"{f:.0%}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(quarters, fontsize=10)
    ax.set_ylabel("Unique Bills", fontsize=11)
    ax.set_title("Y2 Quarter Bill Composition\n(% = carryover rate)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(c + n for c, n in zip(carryover, new_bills)) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panel 2: Cumulative Y1 bill set growth quarter by quarter
    # ------------------------------------------------------------------
    ax2 = axes[1]
    cumul_sizes = [len(y1_cumul[q]) for q in Y1_QUARTER_ORDER]
    qlabels     = [f"2017-{q.upper()}" for q in Y1_QUARTER_ORDER]

    ax2.plot(qlabels, cumul_sizes, marker="o", color=C_LINE, linewidth=2.5,
             markersize=8, zorder=3)
    ax2.fill_between(range(len(qlabels)), cumul_sizes,
                     alpha=0.15, color=C_LINE)

    for i, (lbl, sz) in enumerate(zip(qlabels, cumul_sizes)):
        ax2.text(i, sz + 8, str(sz), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    # Mark total Y1 bill count
    ax2.axhline(len(y1_all), color="gray", linestyle="--", linewidth=1,
                label=f"Y1 total = {len(y1_all)}")
    ax2.set_xticks(range(len(qlabels)))
    ax2.set_xticklabels(qlabels, fontsize=10)
    ax2.set_ylabel("Cumulative Unique Bills", fontsize=11)
    ax2.set_title("Cumulative Y1 Agenda Growth\n(how fast does the bill set stabilise?)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, len(y1_all) * 1.2)
    ax2.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panel 3: Overall Y1 vs Y2 set overlap (proportional area bar)
    # ------------------------------------------------------------------
    ax3 = axes[2]

    total_union = len(y1_all | y2_all)
    segments = [
        (n_y1_only,  "Y1 only",             "#9ecae1"),
        (n_carryover,"Both Y1 & Y2",         C_CARRYOVER),
        (n_new,      "Y2 only (new)",        C_NEW),
    ]
    left = 0
    for val, lbl, color in segments:
        frac = val / total_union
        ax3.barh(0, val, left=left, color=color, height=0.5, label=f"{lbl} ({val})")
        if frac > 0.05:
            ax3.text(left + val / 2, 0, f"{frac:.0%}",
                     ha="center", va="center", fontsize=11,
                     fontweight="bold", color="white")
        left += val

    # Carryover rate annotation
    carryover_rate = n_carryover / len(y2_all)
    ax3.set_xlim(0, total_union * 1.02)
    ax3.set_yticks([])
    ax3.set_xlabel("Unique Bills (union)", fontsize=11)
    ax3.set_title(
        f"Overall Overlap — Y1 ∩ Y2\n"
        f"{carryover_rate:.0%} of Y2 bills already appeared in Y1",
        fontsize=11
    )
    ax3.legend(fontsize=9, loc="upper right")
    ax3.spines[["top", "right", "left"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(results: dict, out_path: Path) -> None:
    """Write plain-text summary of carryover statistics."""
    y1_all   = results["y1_all"]
    y2_all   = results["y2_all"]
    n_c      = len(results["overall_carryover"])
    n_new    = len(results["overall_new"])
    n_y1only = len(results["y1_not_in_y2"])
    y2_rows  = results["y2_rows"]

    lines = [
        "=== 115th Congress — Bill Agenda Carryover (Y1 → Y2) ===\n",
        f"Y1 (2017) unique bills: {len(y1_all)}",
        f"Y2 (2018) unique bills: {len(y2_all)}",
        f"Union: {len(y1_all | y2_all)}",
        f"Intersection (in both): {n_c}  ({n_c/len(y2_all):.1%} of Y2 bills, {n_c/len(y1_all):.1%} of Y1 bills)",
        f"Y2-only (new bills): {n_new}  ({n_new/len(y2_all):.1%} of Y2)",
        f"Y1-only (dropped): {n_y1only}  ({n_y1only/len(y1_all):.1%} of Y1)",
        "",
        "--- Per Y2 quarter ---",
    ]
    for r in y2_rows:
        lines.append(
            f"  {r['quarter']}: {r['total']} bills | "
            f"carryover {r['carryover']} ({r['carryover_frac']:.1%}) | "
            f"new {r['new']} ({1-r['carryover_frac']:.1%})"
        )
    out_path.write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df      = load_data()
    results = run(df)

    png_path = OUT_DIR / "09_agenda_carryover.png"
    txt_path = OUT_DIR / "09_agenda_carryover.txt"

    plot(results, png_path)
    write_report(results, txt_path)

    print(f"Saved: {png_path}")
    print(f"Saved: {txt_path}")
    print(open(txt_path).read())
