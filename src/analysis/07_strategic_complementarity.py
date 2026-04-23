"""
Strategic complementarity in corporate lobbying networks.

Three complementary tests:

  Part A — BCZ payoff complementarity (116th Congress):
    If firm j newly enters bill b at quarter t, does firm i increase its
    lobbying spend on bill b in t+1, and is that response larger when the
    (i,j) RBO edge weight is high?
    Specification:
      Δlog_spend_{i,b,t+1} = β₁ entry_j + β₂ rbo_ij + β₃ (entry_j × rbo_ij)
                            + α_{i,b} + γ_t + ε
    Positive and significant β₃ → BCZ complementarity confirmed.

  Part B — Direction persistence across sessions (111th–117th):
    For decisive pairs (A leads B), is persistence into the next session
    higher for high-RBO pairs than low-RBO pairs?
    Fisher's exact test per consecutive session pair.

  Part C — Direction consistency across all sessions (111th–117th):
    For pairs appearing as decisive in ≥2 sessions, consistency =
    max(n_leads, n_follows) / n_sessions.  Score ∈ [0.5, 1.0] where
    1.0 = same firm always leads, 0.5 = coin-flip.  Mean ≈ 0.77, median = 0.75.

Outputs (outputs/analysis/):
  07_complementarity_regression.csv
  07_persistence_summary.csv
  07_direction_consistency.csv
  07_strategic_complementarity.txt
  07_persistence_bar.png
  07_consistency_hist.png
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import fisher_exact, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESS    = 116
CONGRESSES  = [111, 112, 113, 114, 115, 116, 117]
HIGH_RBO_Q  = 75   # top-quartile for Spec B
LOW_RBO_Q   = 25   # bottom-quartile for Spec C
MAX_Q       = 8    # quarters in 116th Congress

OUT_DIR = ROOT / "outputs" / "analysis"

# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, t):
        for s in self.streams: s.write(t)
    def flush(self):
        for s in self.streams: s.flush()

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_persistence_bar(pers_df, out_dir):
    """Grouped bar chart: high-RBO vs low-RBO persistence rates per congress pair."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    pairs   = pers_df["pair"].tolist()
    hi      = pers_df["high_rbo_persist_rate"].tolist()
    lo      = pers_df["low_rbo_persist_rate"].tolist()
    p_vals  = pers_df["fisher_p"].tolist()
    x       = np.arange(len(pairs))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_hi = ax.bar(x - w/2, hi, w, label="High-RBO (Q4)", color="#C44E52",
                     edgecolor="white", linewidth=0.5)
    bars_lo = ax.bar(x + w/2, lo, w, label="Low-RBO (Q1)",  color="#4C72B0",
                     edgecolor="white", linewidth=0.5)

    # Stars above high-RBO bar for significant Fisher tests
    for xi, p in zip(x, p_vals):
        if p < 0.05:
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            ymax = max(hi[list(x).index(xi)], lo[list(x).index(xi)]) + 0.012
            ax.text(xi, ymax, stars, ha="center", fontsize=11,
                    color="#333333", fontweight="bold")

    ax.axhline(pers_df["persist_rate"].mean(), color="#888888", linewidth=1.2,
               linestyle="--", label=f"Overall mean ({pers_df['persist_rate'].mean():.2%})")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Direction Persistence Rate", fontsize=11)
    ax.set_title("Direction Persistence: High-RBO vs Low-RBO Pairs\n"
                 "111th–117th Congress  (* p<0.05, ** p<0.01, *** p<0.001, Fisher exact)",
                 fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "07_persistence_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consistency_hist(cons_df, out_dir):
    """Histogram of per-pair direction consistency scores (0–1)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    scores = cons_df["consistency"].dropna()
    med    = scores.median()
    mean   = scores.mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores, bins=20, color="#4C72B0", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(med,  color="#C44E52", linewidth=2.0, linestyle="-",
               label=f"Median = {med:.2f}")
    ax.axvline(mean, color="#DD8452", linewidth=1.8, linestyle="--",
               label=f"Mean = {mean:.2f}")
    ax.axvline(0.5, color="#AAAAAA", linewidth=1.2, linestyle=":",
               label="Chance (0.5)")
    ax.set_xlabel("Direction Consistency Score  (fraction of sessions A leads B)",
                  fontsize=11)
    ax.set_ylabel("Number of Pairs", fontsize=11)
    ax.set_title(
        f"Direction Consistency Across Sessions — 111th–117th Congress\n"
        f"(n={len(scores):,} pairs with ≥2 directed sessions; "
        f"{(scores >= 0.8).sum()/len(scores):.0%} ≥ 0.80 consistent)",
        fontsize=11
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "07_consistency_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part A helpers
# ---------------------------------------------------------------------------

def assign_quarters(df):
    """Add 'quarter' column (1–8): 2019 Q1-4 → 1-4, 2020 Q1-4 → 5-8."""
    df = df.copy()
    base_q   = df["report_type"].str[1].astype(int)
    year_off = df["year"].map({2019: 0, 2020: 4})
    df["quarter"] = base_q + year_off
    return df

def build_spend_panel(df_raw):
    """Aggregate (firm, bill, quarter) → total spend."""
    return (
        df_raw.groupby(["fortune_name", "bill_number", "quarter"])["amount_allocated"]
        .sum().reset_index()
        .rename(columns={"fortune_name": "firm", "bill_number": "bill",
                         "amount_allocated": "spend"})
    )

def tag_entry_events(panel):
    """Tag first-quarter entry for each (firm, bill)."""
    first_q = (panel.groupby(["firm", "bill"])["quarter"].min()
               .reset_index().rename(columns={"quarter": "first_quarter"}))
    panel = panel.merge(first_q, on=["firm", "bill"])
    panel["entry_j"] = (panel["quarter"] == panel["first_quarter"]).astype(int)
    return panel.drop(columns=["first_quarter"])

def build_delta_log_spend(panel):
    """Δlog_spend from quarter t to t+1, consecutive quarters, positive spend."""
    srt = panel.sort_values(["firm", "bill", "quarter"]).copy()
    srt["spend_next"]   = srt.groupby(["firm", "bill"])["spend"].shift(-1)
    srt["quarter_next"] = srt.groupby(["firm", "bill"])["quarter"].shift(-1)
    consec = srt[
        (srt["quarter_next"] == srt["quarter"] + 1) &
        (srt["spend"] > 0) & (srt["spend_next"] > 0)
    ].copy()
    consec["delta_log_spend"] = np.log(consec["spend_next"]) - np.log(consec["spend"])
    return consec[["firm", "bill", "quarter", "delta_log_spend"]].rename(
        columns={"firm": "firm_i"})

def build_rbo_lookup(rbo_df):
    """Symmetric (i,j) and (j,i) RBO weight lookup."""
    sym = pd.concat([
        rbo_df[["source", "target", "rbo"]].rename(
            columns={"source": "firm_i", "target": "firm_j"}),
        rbo_df[["source", "target", "rbo"]].rename(
            columns={"target": "firm_i", "source": "firm_j"}),
    ], ignore_index=True)
    return sym

def build_regression_panel(delta_df, panel_with_entry, rbo_sym):
    """Cross (firm_i, bill, quarter) with active firm_j; join RBO weights."""
    j_status = panel_with_entry[["firm", "bill", "quarter", "entry_j"]].rename(
        columns={"firm": "firm_j"})
    merged = delta_df.merge(j_status, on=["bill", "quarter"], how="inner")
    merged = merged[merged["firm_i"] != merged["firm_j"]]
    merged = merged.merge(rbo_sym, on=["firm_i", "firm_j"], how="inner")
    merged["rbo_ij"]     = merged["rbo"]
    merged["entry_x_rbo"] = merged["entry_j"] * merged["rbo_ij"]
    merged["firm_bill"]   = merged["firm_i"] + "||" + merged["bill"]
    return merged.drop(columns=["rbo"]).reset_index(drop=True)

def demean_within(df, group_col, value_cols):
    """Subtract group mean (within-transformation to absorb group FE)."""
    df = df.copy()
    gm = df.groupby(group_col)[value_cols].transform("mean")
    for c in value_cols:
        df[c + "_dm"] = df[c] - gm[c]
    return df

def run_ols_spec(df, label):
    """Within-transformed OLS with quarter FE and HC3 SE."""
    cols_to_demean = ["delta_log_spend", "entry_j", "rbo_ij", "entry_x_rbo"]
    df_dm = demean_within(df, "firm_bill", cols_to_demean)

    q_dummies = pd.get_dummies(df["quarter"], prefix="q", drop_first=True)
    for col in q_dummies.columns:
        q_dummies[col] = q_dummies[col].astype(float)
        gm = q_dummies[col].groupby(df["firm_bill"]).transform("mean")
        df_dm[col + "_dm"] = q_dummies[col] - gm

    q_dm_cols = [c for c in df_dm.columns if c.endswith("_dm") and c.startswith("q")]
    X_cols    = ["entry_j_dm", "rbo_ij_dm", "entry_x_rbo_dm"] + q_dm_cols
    X = df_dm[X_cols].copy()
    X.insert(0, "const", 0.0)
    y = df_dm["delta_log_spend_dm"]

    mask = df.groupby("firm_bill")["firm_bill"].transform("count") > 1
    result = sm.OLS(y[mask], X[mask]).fit(cov_type="HC3")
    return result, int(mask.sum())

# ---------------------------------------------------------------------------
# Part A: BCZ payoff complementarity
# ---------------------------------------------------------------------------

def run_part_a():
    """Run payoff complementarity regression on 116th Congress."""
    df_raw = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/opensecrets_lda_reports.csv")
    df_raw = assign_quarters(df_raw)
    rbo    = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/rbo_directed_influence.csv")

    panel            = build_spend_panel(df_raw)
    panel_with_entry = tag_entry_events(panel)
    delta_df         = build_delta_log_spend(panel)
    rbo_sym          = build_rbo_lookup(rbo)
    main_panel       = build_regression_panel(delta_df, panel_with_entry, rbo_sym)

    rbo_q75 = np.percentile(main_panel["rbo_ij"], HIGH_RBO_Q)
    rbo_q25 = np.percentile(main_panel["rbo_ij"], LOW_RBO_Q)

    print(f"\n  116th Congress panel: {len(main_panel):,} obs  "
          f"| {main_panel['firm_i'].nunique()} firm_i  "
          f"| {main_panel['firm_bill'].nunique()} firm-bill groups")
    print(f"  RBO p25={rbo_q25:.4f}  p75={rbo_q75:.4f}")

    results = []
    specs   = [
        ("A — full RBO-linked",   main_panel),
        (f"B — high-RBO (≥p75)",  main_panel[main_panel["rbo_ij"] >= rbo_q75].copy()),
        (f"C — low-RBO (<p25)",   main_panel[main_panel["rbo_ij"] <  rbo_q25].copy()),
    ]

    print(f"\n  {'Spec':<28} {'N':>8} {'β₁(entry)':>12} {'β₃(inter)':>12} "
          f"{'SE(β₃)':>10} {'p(β₃)':>10} {'Interp'}")
    print(f"  {'─'*94}")
    for label, df_spec in specs:
        if len(df_spec) < 50:
            print(f"  {label:<28} (insufficient obs — skipped)")
            continue
        res, n = run_ols_spec(df_spec, label)
        b1  = res.params.get("entry_j_dm", np.nan)
        b3  = res.params.get("entry_x_rbo_dm", np.nan)
        se3 = res.bse.get("entry_x_rbo_dm", np.nan)
        p3  = res.pvalues.get("entry_x_rbo_dm", np.nan)
        stars = "***" if p3 < 0.001 else "**" if p3 < 0.01 else "*" if p3 < 0.05 else ""
        interp = ("complementarity" if b3 > 0 and p3 < 0.05
                  else "ns" if p3 >= 0.05 else "negative")
        print(f"  {label:<28} {n:>8,} {float(b1):>12.4f} {float(b3):>12.4f} "
              f"{float(se3):>10.4f} {float(p3):>10.4f}{stars}  {interp}")
        results.append({
            "spec": label, "n": n,
            "coef_entry_j": round(float(b1), 5),
            "coef_entry_x_rbo": round(float(b3), 5),
            "se_entry_x_rbo": round(float(se3), 5),
            "p_entry_x_rbo": round(float(p3), 5),
            "r2": round(res.rsquared, 5),
        })
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Part B: Direction persistence
# ---------------------------------------------------------------------------

def run_part_b():
    """Test whether high-RBO pairs persist in direction more across sessions."""
    persistence_rows = []

    for i in range(len(CONGRESSES) - 1):
        ci, cj = CONGRESSES[i], CONGRESSES[i + 1]
        pi = DATA_DIR / f"congress/{ci}/rbo_directed_influence.csv"
        pj = DATA_DIR / f"congress/{cj}/rbo_directed_influence.csv"
        if not pi.exists() or not pj.exists():
            continue

        ei = pd.read_csv(pi)
        ej = pd.read_csv(pj)
        if "rbo" not in ei.columns or "rbo" not in ej.columns:
            continue

        # Canonical decisive pairs in ci (source < target, source leads)
        dec_i = ei[(ei["source"] < ei["target"]) & (ei["net_temporal"] > 0)][
            ["source", "target", "rbo", "net_temporal"]
        ].copy()
        dec_i.columns = ["source", "target", "rbo_n", "nt_n"]

        # Look up in cj
        canon_j = ej[ej["source"] < ej["target"]].copy()
        lookup  = (
            canon_j.set_index(["source", "target"])[["net_temporal"]]
            .rename(columns={"net_temporal": "nt_n1"})
        )
        merged = dec_i.join(lookup, on=["source", "target"], how="left")
        merged["persists"] = (merged["nt_n1"] > 0).astype("Int8")

        # RBO quartile split
        merged["rbo_quartile"] = pd.qcut(
            merged["rbo_n"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        )

        # Fisher's exact: Q4 vs Q1 × persists
        high = merged[merged["rbo_quartile"] == "Q4 (high)"].dropna(subset=["nt_n1"])
        low  = merged[merged["rbo_quartile"] == "Q1 (low)"].dropna(subset=["nt_n1"])

        if len(high) < 3 or len(low) < 3:
            continue

        ct = np.array([
            [int(high["persists"].sum()), len(high) - int(high["persists"].sum())],
            [int(low["persists"].sum()),  len(low)  - int(low["persists"].sum())],
        ])
        _, p_fisher = fisher_exact(ct, alternative="greater")

        n_dec   = len(merged)
        n_in_nj = int(merged["nt_n1"].notna().sum())
        n_pers  = int(merged["persists"].sum(skipna=True))

        print(f"  {ci}→{cj}:  decisive={n_dec:,}  in_next={n_in_nj:,}  "
              f"persist={n_pers:,}  "
              f"({100*n_pers/max(n_in_nj,1):.1f}%)  |  "
              f"High-RBO persist: {high['persists'].mean():.3f}  "
              f"Low-RBO persist: {low['persists'].mean():.3f}  "
              f"Fisher p={p_fisher:.4f}")

        persistence_rows.append({
            "pair": f"{ci}->{cj}",
            "n_decisive": n_dec, "n_in_next": n_in_nj, "n_persist": n_pers,
            "persist_rate": round(n_pers / max(n_in_nj, 1), 4),
            "high_rbo_persist_rate": round(float(high["persists"].mean(skipna=True)), 4),
            "low_rbo_persist_rate":  round(float(low["persists"].mean(skipna=True)), 4),
            "fisher_p": round(p_fisher, 5),
        })

    return pd.DataFrame(persistence_rows)

# ---------------------------------------------------------------------------
# Part C: Direction consistency across all sessions
# ---------------------------------------------------------------------------

def run_part_c():
    """
    Per-pair direction consistency: fraction of multi-session decisive appearances
    where the same firm leads.  Pairs with ≥2 directed sessions only.
    """
    # Collect canonical decisive records (source < target, net_temporal != 0)
    # across all seven congresses
    all_records = []
    for c in CONGRESSES:
        path = DATA_DIR / f"congress/{c}/rbo_directed_influence.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "net_temporal" not in df.columns:
            continue
        dec = df[(df["source"] < df["target"]) & (df["net_temporal"] != 0)][
            ["source", "target", "net_temporal"]
        ].copy()
        dec["congress"] = c
        all_records.append(dec)

    if not all_records:
        return pd.DataFrame()

    records = pd.concat(all_records, ignore_index=True)
    records["a_leads"] = (records["net_temporal"] > 0).astype(int)

    # Aggregate per canonical pair
    agg = (
        records.groupby(["source", "target"])
        .agg(n_sessions=("congress", "count"),
             n_leads=("a_leads", "sum"))
        .reset_index()
    )
    # Keep pairs with ≥2 directed sessions
    agg = agg[agg["n_sessions"] >= 2].copy()
    agg["n_follows"] = agg["n_sessions"] - agg["n_leads"]
    # Consistency = fraction of sessions the MAJORITY-DIRECTION firm leads.
    # Takes max over both directions so the score lives in [0.5, 1.0].
    # A score of 1.0 means the same firm always leads; 0.5 means coin-flip.
    agg["consistency"] = agg[["n_leads", "n_follows"]].max(axis=1) / agg["n_sessions"]
    agg = agg.sort_values("consistency", ascending=False).reset_index(drop=True)

    n        = len(agg)
    mean_c   = agg["consistency"].mean()
    med_c    = agg["consistency"].median()
    pct_80   = (agg["consistency"] >= 0.80).sum() / n
    pct_100  = (agg["consistency"] == 1.00).sum() / n
    pct_maj  = (agg["consistency"] > 0.50).sum() / n

    print(f"\n  Pairs with ≥2 directed sessions: {n:,}")
    print(f"  Mean consistency:   {mean_c:.4f}")
    print(f"  Median consistency: {med_c:.4f}")
    print(f"  Pairs ≥ 0.80 consistent: {int(pct_80*n):,} / {n:,}  ({pct_80:.1%})")
    print(f"  Perfectly consistent (1.00): {int(pct_100*n):,} / {n:,}  ({pct_100:.1%})")
    print(f"  Pairs with majority-direction (>0.5): {int(pct_maj*n):,} / {n:,}  ({pct_maj:.1%})")

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "07_strategic_complementarity.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 07: STRATEGIC COMPLEMENTARITY")
    print(SEP)

    # -- Part A ----------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART A: BCZ PAYOFF COMPLEMENTARITY (116th Congress)")
    print(f"{'─'*70}")
    print("""
  Specification:
    Δlog_spend_{i,b,t+1} = β₁ entry_j + β₂ rbo_ij + β₃ (entry_j × rbo_ij)
                          + firm-bill FE + quarter FE + ε
  Positive, significant β₃ → spending increases more in response to a
  co-lobbyist's entry when the pair has higher RBO similarity (BCZ-style
  strategic complementarity).
""")
    reg_df = run_part_a()
    reg_df.to_csv(OUT_DIR / "07_complementarity_regression.csv", index=False)

    # -- Part B ----------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART B: DIRECTION PERSISTENCE BY RBO QUARTILE (111th–117th)")
    print(f"{'─'*70}")
    print("""
  For each consecutive congress pair, decisive pairs (A leads B in session N)
  are split by RBO quartile. Fisher's exact test (one-sided):
  H1: high-RBO pairs (Q4) persist significantly more than low-RBO pairs (Q1).
""")
    pers_df = run_part_b()

    if not pers_df.empty:
        n_sig = (pers_df["fisher_p"] < 0.05).sum()
        print(f"\n  Summary (Part B):")
        print(f"  Mean overall persistence rate:        {pers_df['persist_rate'].mean():.3f}")
        print(f"  Mean high-RBO (Q4) persistence rate:  {pers_df['high_rbo_persist_rate'].mean():.3f}")
        print(f"  Mean low-RBO (Q1) persistence rate:   {pers_df['low_rbo_persist_rate'].mean():.3f}")
        diff = pers_df['high_rbo_persist_rate'].mean() - pers_df['low_rbo_persist_rate'].mean()
        print(f"  Mean differential (high − low):       {diff:+.3f}")
        print(f"  Fisher p < 0.05:  {n_sig}/{len(pers_df)} consecutive pairs")
        print(f"  → {'Consistent with complementarity' if diff > 0 and n_sig > 0 else 'Weak or no complementarity differential'}")
        pers_df.to_csv(OUT_DIR / "07_persistence_summary.csv", index=False)
        plot_persistence_bar(pers_df, OUT_DIR)

    # -- Part C ----------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART C: DIRECTION CONSISTENCY ACROSS ALL SESSIONS (111th–117th)")
    print(f"{'─'*70}")
    print("""
  For each canonical pair (A, B) appearing as decisive in ≥2 sessions,
  consistency = max(n_leads, n_follows) / n_sessions ∈ [0.5, 1.0].
  1.0 = same firm always leads; 0.5 = coin-flip.
  This differs from Part B: it asks whether direction is stable across all
  sessions a pair co-occurs, not just between consecutive session pairs.
  (Part B: ~33% of decisive pairs survive into the next specific session;
   Part C: among pairs with ≥2 appearances, direction is consistent 77% of the time.)
""")
    cons_df = run_part_c()
    if not cons_df.empty:
        cons_df.to_csv(OUT_DIR / "07_direction_consistency.csv", index=False)
        plot_consistency_hist(cons_df, OUT_DIR)

    print(f"\n  Outputs:")
    print(f"    07_complementarity_regression.csv")
    print(f"    07_persistence_summary.csv")
    print(f"    07_direction_consistency.csv")
    print(f"    07_strategic_complementarity.txt")
    print(f"    07_persistence_bar.png")
    print(f"    07_consistency_hist.png")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
