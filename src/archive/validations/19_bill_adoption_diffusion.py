"""
Bill adoption diffusion test — agenda-setting via follower adoption (116th Congress).

Question: Is a net follower B, connected to influencer A via a directed RBO edge,
more likely to adopt a bill that A has previously lobbied, compared to B's with
weak or no connections to A? Does that effect persist over Q+1, Q+2, Q+3?

Setup:
  For each directed (A→B) pair in the aggregate network, identify all bills X
  where A enters in quarter t and B has not yet entered X at or before t.
  The outcome is whether B first enters X within k quarters after A's entry.

  The unit of observation is (A, B, bill, a_entry_quarter) — one row per
  "exposure event" where B is a potential follower on a bill A just adopted.

Analysis:
  Part 1 — Descriptive adoption rates by RBO quartile × horizon (Q+1/Q+2/Q+3).
            Horizon k restricted to cases where a_entry_quarter + k ≤ 8.

  Part 2 — Logit regression (LPM robustness): adoption within Q+1, Q+2, Q+3
            as a function of rbo_weight, controlling for:
              - log(n_firms_on_bill): bill popularity (how many firms lobby it)
              - a_net_influence: A's overall influence score (stronger A → more signal)
              - b_net_influence: B's overall followership tendency
              - a_entry_quarter: when A entered (early vs. late bills)
              - bill FE (in LPM) or absorbed via controls in Logit
            Key coefficient: rbo_weight (or log-rbo for interpretability)

  Part 3 — Adoption curve: cumulative adoption rate at Q+1, Q+2, Q+3 for
            high-RBO (≥ median) vs. low-RBO (<median) pairs, with 95% CI.
            Also: adoption curve split by A's quartile of net_influence.

  Part 4 — Robustness: restrict to bills X where A is the UNIQUE first-mover
            (no other firm enters X in the same quarter as A). Cleaner
            attribution — B is responding to A specifically, not to a wave.

Identification note:
  The comparison is within-bill: for a given bill X entered by A at quarter t,
  high-RBO followers are compared to low-RBO followers of A on that same bill.
  Bill FE (or log(n_firms) control) absorbs cross-bill variation in salience.
  The remaining variation is the RBO weight of the A→B link.

Run from src/ directory:
  python validations/19_bill_adoption_diffusion.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import fisher_exact, norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR   = ROOT / "outputs" / "validation"
FIG_DIR   = ROOT / "visualizations" / "png"
TXT_PATH  = OUT_DIR / "19_bill_adoption_diffusion.txt"
CSV_CAND  = OUT_DIR / "19_adoption_candidates.csv"
CSV_RATES = OUT_DIR / "19_adoption_rates.csv"
CSV_REG   = OUT_DIR / "19_adoption_regression.csv"

HORIZONS  = [1, 2, 3]   # quarters ahead to track adoption
MAX_Q     = 8            # last quarter of 116th Congress


# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def assign_quarters(df):
    """Add 'quarter' column (1–8): 2019 Q1-4 → 1-4, 2020 Q1-4 → 5-8."""
    df = df.copy()
    base_q   = df["report_type"].str[1].astype(int)
    year_off = df["year"].map({2019: 0, 2020: 4})
    df["quarter"] = base_q + year_off
    return df


def build_first_quarter_table(df_raw):
    """
    Build (firm, bill, first_q): the first quarter each firm lobbied each bill.
    Only rows with positive allocated spend are counted.
    """
    panel = (
        df_raw.groupby(["fortune_name", "bill_number", "quarter"])["amount_allocated"]
        .sum()
        .reset_index()
        .rename(columns={"fortune_name": "firm", "bill_number": "bill",
                         "amount_allocated": "spend"})
    )
    panel = panel[panel["spend"] > 0]
    first_q = (
        panel.groupby(["firm", "bill"])["quarter"]
        .min()
        .reset_index()
        .rename(columns={"quarter": "first_q"})
    )
    # Bill-level stats: total number of unique firms ever lobbying the bill
    bill_firms = panel.groupby("bill")["firm"].nunique().rename("n_firms_on_bill")
    return first_q, bill_firms


def build_candidate_set(first_q_df, directed_df, bill_firms, node_attrs):
    """
    Construct the candidate set: one row per (A, B, bill) where:
      - A→B is a directed edge (A is net influencer, B is net follower)
      - A first lobbies bill X at quarter t (a_entry_q)
      - B had not lobbied bill X at or before quarter t

    Columns added:
      - b_first_q:    first quarter B ever lobbied bill X (NaN if never)
      - adopted_q1/q2/q3: binary, B first enters within k quarters of A
      - horizon_obs_q1/q2/q3: 1 if a_entry_q + k ≤ MAX_Q (adoption observable)
      - n_firms_on_bill: total firms ever on that bill (popularity proxy)
      - a_net_influence, b_net_influence: from node_attributes.csv
    """
    # A's entry events
    a_first = first_q_df.rename(columns={"firm": "A", "first_q": "a_entry_q"})

    # Cross directed pairs with A's bill entries
    cand = directed_df.merge(a_first, on="A")   # (A, B, rbo_weight, bill, a_entry_q)

    # Join B's first quarter on the same bill
    b_first = first_q_df.rename(columns={"firm": "B", "first_q": "b_first_q"})
    cand = cand.merge(b_first, on=["B", "bill"], how="left")

    # Keep only rows where B had not yet entered when A did
    b_not_yet = (
        cand["b_first_q"].isna() |
        (cand["b_first_q"] > cand["a_entry_q"])
    )
    cand = cand[b_not_yet].copy()

    # Adoption indicators and horizon observability
    for k in HORIZONS:
        cand[f"adopted_q{k}"] = (
            cand["b_first_q"].notna() &
            (cand["b_first_q"] <= cand["a_entry_q"] + k)
        ).astype(int)
        cand[f"horizon_obs_q{k}"] = (cand["a_entry_q"] + k <= MAX_Q).astype(int)

    # Bill popularity
    cand = cand.merge(bill_firms, on="bill", how="left")

    # Node-level influence scores
    a_attr = node_attrs[["firm", "net_influence"]].rename(
        columns={"firm": "A", "net_influence": "a_net_influence"})
    b_attr = node_attrs[["firm", "net_influence"]].rename(
        columns={"firm": "B", "net_influence": "b_net_influence"})
    cand = cand.merge(a_attr, on="A", how="left")
    cand = cand.merge(b_attr, on="B", how="left")

    # Log transforms for regression
    cand["log_rbo"]        = np.log(cand["rbo_weight"] + 1e-8)
    cand["log_n_firms"]    = np.log(cand["n_firms_on_bill"].clip(lower=1))
    cand["a_entry_q_norm"] = cand["a_entry_q"] / MAX_Q

    return cand.reset_index(drop=True)


def build_unique_entry_subset(cand, first_q_df):
    """
    Restrict to bills where A is the sole first-mover (no other firm enters
    bill X in the same quarter as A). Cleaner causal attribution.
    """
    # Count how many firms first-entered each bill in each quarter
    entry_counts = (
        first_q_df.groupby(["bill", "first_q"])["firm"]
        .nunique()
        .reset_index()
        .rename(columns={"first_q": "a_entry_q", "firm": "n_coentrants"})
    )
    cand2 = cand.merge(entry_counts, on=["bill", "a_entry_q"], how="left")
    cand2["n_coentrants"] = cand2["n_coentrants"].fillna(1)
    # Subtract A itself
    return cand2[cand2["n_coentrants"] == 1].copy()


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def adoption_rate_ci(adopted, n, z=1.96):
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = adopted / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(p, 5), round(centre - half, 5), round(centre + half, 5)


def adoption_table(cand, label="Full"):
    """Compute adoption rates at Q+1, Q+2, Q+3 by RBO quartile."""
    cand = cand.copy()
    cand["rbo_quartile"] = pd.qcut(cand["rbo_weight"], q=4,
                                   labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    rows = []
    for k in HORIZONS:
        obs = cand[cand[f"horizon_obs_q{k}"] == 1]
        overall_p, ci_lo, ci_hi = adoption_rate_ci(
            obs[f"adopted_q{k}"].sum(), len(obs))
        rows.append({
            "sample": label, "horizon": f"Q+{k}", "rbo_group": "All",
            "n": len(obs), "n_adopted": int(obs[f"adopted_q{k}"].sum()),
            "rate": overall_p, "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
        for qg in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
            sub = obs[obs["rbo_quartile"] == qg]
            p, lo, hi = adoption_rate_ci(sub[f"adopted_q{k}"].sum(), len(sub))
            rows.append({
                "sample": label, "horizon": f"Q+{k}", "rbo_group": qg,
                "n": len(sub), "n_adopted": int(sub[f"adopted_q{k}"].sum()),
                "rate": p, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def print_adoption_table(df, header):
    """Print adoption rate table grouped by horizon."""
    print(f"\n  {header}")
    print(f"  {'Horizon':<8} {'RBO group':<14} {'N':>7} {'Adopted':>8} {'Rate':>8} {'95% CI'}")
    print(f"  {'-'*65}")
    for horizon in [f"Q+{k}" for k in HORIZONS]:
        sub = df[df["horizon"] == horizon]
        for _, row in sub.iterrows():
            lo = f"{row['ci_lo']:.4f}" if pd.notna(row["ci_lo"]) else "—"
            hi = f"{row['ci_hi']:.4f}" if pd.notna(row["ci_hi"]) else "—"
            print(f"  {row['horizon']:<8} {row['rbo_group']:<14} {int(row['n']):>7} "
                  f"{int(row['n_adopted']):>8} {row['rate']:>8.4f}  [{lo}, {hi}]")


def run_logit(cand, k, label):
    """
    Logit and LPM (OLS) for P(adopted within Q+k) ~ log_rbo + log_n_firms
    + a_net_influence + b_net_influence + a_entry_q_norm.
    Restricts to observable-horizon rows. Drops rows with any NaN covariate.
    Returns (logit_result, lpm_result, n_used).
    """
    obs = cand[cand[f"horizon_obs_q{k}"] == 1].copy()
    covars = ["log_rbo", "log_n_firms", "a_net_influence",
              "b_net_influence", "a_entry_q_norm"]
    obs = obs.dropna(subset=[f"adopted_q{k}"] + covars)

    y = obs[f"adopted_q{k}"].astype(float)
    X = sm.add_constant(obs[covars].astype(float))

    try:
        logit_res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    except Exception as e:
        logit_res = None
        print(f"    Logit failed for {label} Q+{k}: {e}")

    lpm_res = sm.OLS(y, X).fit(cov_type="HC3")
    return logit_res, lpm_res, len(obs)


def print_reg_table(logit_res, lpm_res, n, label, k):
    """Print compact regression table for key coefficients."""
    print(f"\n  {label} — Q+{k} adoption (N={n:,})")
    print(f"  {'Variable':<24} {'Logit coef':>12} {'Logit p':>9} {'LPM coef':>12} {'LPM p':>9}")
    print(f"  {'-'*70}")
    display_vars = {
        "log_rbo":         "log(rbo_weight)",
        "log_n_firms":     "log(n_firms_bill)",
        "a_net_influence": "A net_influence",
        "b_net_influence": "B net_influence",
        "a_entry_q_norm":  "A entry quarter",
    }
    for var, label_v in display_vars.items():
        lpm_b = lpm_res.params.get(var, np.nan)
        lpm_p = lpm_res.pvalues.get(var, np.nan)
        lpm_s = "***" if lpm_p < 0.001 else "**" if lpm_p < 0.01 else "*" if lpm_p < 0.05 else ""
        if logit_res is not None and var in logit_res.params.index:
            lo_b = logit_res.params[var]
            lo_p = logit_res.pvalues[var]
            lo_s = "***" if lo_p < 0.001 else "**" if lo_p < 0.01 else "*" if lo_p < 0.05 else ""
            print(f"  {label_v:<24} {lo_b:>12.4f}{lo_s:<3} {lo_p:>9.4f} "
                  f"{lpm_b:>12.4f}{lpm_s:<3} {lpm_p:>9.4f}")
        else:
            print(f"  {label_v:<24} {'—':>15} {'—':>9} "
                  f"{lpm_b:>12.4f}{lpm_s:<3} {lpm_p:>9.4f}")


def extract_reg_row(logit_res, lpm_res, n, sample, k):
    """Extract key result fields for CSV export."""
    row = {"sample": sample, "horizon": f"Q+{k}", "n": n}
    for var in ["log_rbo", "log_n_firms", "a_net_influence", "b_net_influence"]:
        row[f"lpm_coef_{var}"]  = round(lpm_res.params.get(var, np.nan), 5)
        row[f"lpm_p_{var}"]     = round(lpm_res.pvalues.get(var, np.nan), 5)
        if logit_res is not None and var in logit_res.params.index:
            row[f"logit_coef_{var}"] = round(logit_res.params[var], 5)
            row[f"logit_p_{var}"]    = round(logit_res.pvalues[var], 5)
    return row


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_adoption_curve(rates_full, rates_unique, out_path):
    """
    Two-panel figure:
      Left:  cumulative adoption rate at Q+1/2/3 for high vs. low RBO
             (full sample vs. unique-entry robustness)
      Right: Q+1 adoption rate by RBO quartile, full sample
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: adoption curve high vs low RBO ---
    ax = axes[0]
    horizons_x = [1, 2, 3]

    for sample_label, rates_df, style in [
        ("Full — High RBO (≥ median)", rates_full,   {"color": "#2196F3", "ls": "-",  "marker": "o"}),
        ("Full — Low RBO (< median)",  rates_full,   {"color": "#F44336", "ls": "-",  "marker": "s"}),
        ("Unique entry — High RBO",    rates_unique, {"color": "#2196F3", "ls": "--", "marker": "o"}),
        ("Unique entry — Low RBO",     rates_unique, {"color": "#F44336", "ls": "--", "marker": "s"}),
    ]:
        is_high = "High" in sample_label
        grp = "Q4 (high)" if is_high else "Q1 (low)"
        # Use Q4 as proxy for ≥median and Q1 as proxy for <median here; actual
        # high/low split computed separately below
        pass  # will plot from precomputed high_low arrays

    # Compute high/low RBO adoption rates (median split, not quartile)
    for cand_df, style_hi, style_lo, lbl_suffix in [
        (None, None, None, None),   # placeholder; computed below
    ]:
        pass

    ax.set_title("Adoption curve: High vs. Low RBO (median split)", fontsize=11)
    ax.set_xlabel("Quarters after A's entry on bill")
    ax.set_ylabel("Cumulative adoption rate")
    ax.set_xticks([1, 2, 3])

    # --- Right panel: Q+1 adoption by RBO quartile ---
    ax2 = axes[1]

    q1_rates = rates_full[
        (rates_full["horizon"] == "Q+1") &
        (rates_full["rbo_group"].isin(["Q1 (low)", "Q2", "Q3", "Q4 (high)"]))
    ].copy()

    colors = ["#EF9A9A", "#FFCC80", "#A5D6A7", "#64B5F6"]
    bars = ax2.bar(q1_rates["rbo_group"], q1_rates["rate"],
                   color=colors, edgecolor="white", width=0.6)
    ax2.errorbar(
        q1_rates["rbo_group"],
        q1_rates["rate"],
        yerr=[q1_rates["rate"] - q1_rates["ci_lo"],
              q1_rates["ci_hi"] - q1_rates["rate"]],
        fmt="none", color="black", capsize=4, lw=1.5
    )
    for bar, (_, row) in zip(bars, q1_rates.iterrows()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{row['rate']:.3f}\n(n={int(row['n']):,})",
                 ha="center", va="bottom", fontsize=8)

    ax2.set_title("Q+1 adoption rate by RBO quartile (full sample)", fontsize=11)
    ax2.set_xlabel("RBO weight quartile")
    ax2.set_ylabel("Adoption rate within Q+1")
    ax2.set_ylim(0, max(q1_rates["ci_hi"]) * 1.25)
    ax2.grid(axis="y", alpha=0.3)

    # Now fill the left panel properly using precomputed median-split rates
    fig.delaxes(axes[0])
    ax_l = fig.add_subplot(1, 2, 1)

    # We need to pass precomputed median-split data — rebuild from rates_full
    # using Q4 as high, Q1 as low (quartile-level proxy)
    for grp, color, marker, ls, lbl in [
        ("Q4 (high)", "#2196F3", "o", "-",  "High RBO — full sample"),
        ("Q1 (low)",  "#F44336", "s", "-",  "Low RBO — full sample"),
        ("Q4 (high)", "#2196F3", "o", "--", "High RBO — unique entry"),
        ("Q1 (low)",  "#F44336", "s", "--", "Low RBO — unique entry"),
    ]:
        src = rates_full if "full" in lbl else rates_unique
        pts = src[(src["rbo_group"] == grp)].sort_values("horizon")
        xs = [int(h[-1]) for h in pts["horizon"]]
        ys = pts["rate"].values
        los = pts["ci_lo"].values
        his = pts["ci_hi"].values
        mask = (src["sample"] == src["sample"].iloc[0])  # all rows from same df
        ax_l.plot(xs, ys, color=color, ls=ls, marker=marker, ms=7, lw=2, label=lbl)
        ax_l.fill_between(xs, los, his, color=color, alpha=0.1)

    ax_l.set_title("Adoption curve: Q1/Q4 RBO groups × sample", fontsize=11)
    ax_l.set_xlabel("Quarters after A's entry on bill")
    ax_l.set_ylabel("Cumulative adoption rate")
    ax_l.set_xticks([1, 2, 3])
    ax_l.legend(fontsize=8, loc="upper left")
    ax_l.grid(alpha=0.3)

    fig.suptitle(
        "Bill Adoption Diffusion — High-RBO Followers vs. Low-RBO Followers\n"
        "116th Congress (A→B directed edges, Q+1 through Q+3)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 72)
    print("VALIDATION 19: BILL ADOPTION DIFFUSION (116th CONGRESS)")
    print("=" * 72)
    print()
    print("Question: Is a net follower B connected to influencer A more likely")
    print("to adopt bill X that A lobbied, and does high RBO amplify that?")
    print(f"Horizons: Q+1, Q+2, Q+3 (a_entry_q + k ≤ {MAX_Q})")
    print()

    # -- Load data -----------------------------------------------------------
    print("[1/5] Building candidate set ...")
    df_raw = pd.read_csv(
        DATA_DIR / "congress" / "116" / "opensecrets_lda_reports.csv"
    )
    df_raw = assign_quarters(df_raw)

    rbo = pd.read_csv(
        DATA_DIR / "congress" / "116" / "rbo_directed_influence.csv"
    )
    node_attrs = pd.read_csv(
        DATA_DIR / "congress" / "116" / "node_attributes.csv"
    )

    # Decisive edges only (net_temporal > 0): A is the net first-mover over B.
    # Use 'rbo' column (full RBO similarity) as the structural weight for regression.
    directed = (
        rbo[rbo["net_temporal"] > 0][["source", "target", "rbo"]]
        .rename(columns={"source": "A", "target": "B", "rbo": "rbo_weight"})
        .copy()
    )
    print(f"  Directed (A→B) edges:   {len(directed):,}")
    print(f"  A firms:                {directed['A'].nunique()}")
    print(f"  B firms:                {directed['B'].nunique()}")

    first_q_df, bill_firms = build_first_quarter_table(df_raw)
    print(f"  (firm, bill) entries:   {len(first_q_df):,}")
    print(f"  Bills with any lobbying:{bill_firms.shape[0]:,}")

    cand = build_candidate_set(first_q_df, directed, bill_firms, node_attrs)
    print(f"\n  Candidate (A, B, bill) rows:       {len(cand):,}")
    print(f"  A firms in candidates:             {cand['A'].nunique()}")
    print(f"  B firms in candidates:             {cand['B'].nunique()}")
    print(f"  Unique bills:                      {cand['bill'].nunique()}")
    print(f"  Unique (A, B) pairs:               {cand.groupby(['A','B']).ngroups}")
    for k in HORIZONS:
        n_obs = cand[f"horizon_obs_q{k}"].sum()
        print(f"  Observable at Q+{k} horizon:        {n_obs:,}")

    # Unique-entry robustness subset
    cand_unique = build_unique_entry_subset(cand, first_q_df)
    print(f"\n  Unique-entry subset (A sole first-mover on bill): {len(cand_unique):,}")
    for k in HORIZONS:
        n_obs = cand_unique[f"horizon_obs_q{k}"].sum()
        print(f"  Observable at Q+{k} (unique entry):  {n_obs:,}")

    cand.to_csv(CSV_CAND, index=False)
    print(f"\n  Candidate CSV → {CSV_CAND.name}")

    # -- Part 1: Adoption rates by RBO quartile ------------------------------
    print("\n[2/5] Adoption rates by RBO quartile and horizon ...")
    rates_full   = adoption_table(cand,        label="Full sample")
    rates_unique = adoption_table(cand_unique, label="Unique entry")

    print_adoption_table(rates_full,   "Full sample")
    print_adoption_table(rates_unique, "Unique-entry robustness")

    # Summary: high vs low RBO (median split)
    rbo_med = cand["rbo_weight"].median()
    print(f"\n  RBO weight median (directed edges): {rbo_med:.4f}")
    print(f"\n  Median-split adoption rates (full sample):")
    print(f"  {'Horizon':<8} {'High RBO ≥med':>16} {'Low RBO <med':>16} {'Ratio':>8}")
    print(f"  {'-'*52}")
    for k in HORIZONS:
        obs = cand[cand[f"horizon_obs_q{k}"] == 1]
        hi  = obs[obs["rbo_weight"] >= rbo_med]
        lo  = obs[obs["rbo_weight"] < rbo_med]
        r_hi = hi[f"adopted_q{k}"].mean()
        r_lo = lo[f"adopted_q{k}"].mean()
        ratio = r_hi / r_lo if r_lo > 0 else np.nan
        print(f"  Q+{k}     {r_hi:>10.4f} (n={len(hi):,})  "
              f"{r_lo:>10.4f} (n={len(lo):,})  {ratio:>8.3f}x")

    # Fisher's exact test at Q+1 (2×2 table)
    obs1 = cand[cand["horizon_obs_q1"] == 1]
    hi1  = obs1[obs1["rbo_weight"] >= rbo_med]
    lo1  = obs1[obs1["rbo_weight"] <  rbo_med]
    ct = np.array([
        [hi1["adopted_q1"].sum(),   len(hi1) - hi1["adopted_q1"].sum()],
        [lo1["adopted_q1"].sum(),   len(lo1) - lo1["adopted_q1"].sum()],
    ])
    from scipy.stats import chi2_contingency
    chi2, p_chi, _, _ = chi2_contingency(ct)
    print(f"\n  χ² test (high vs low RBO, Q+1): χ²={chi2:.3f}, p={p_chi:.4f}")

    # Save rates
    all_rates = pd.concat([rates_full, rates_unique], ignore_index=True)
    all_rates.to_csv(CSV_RATES, index=False)
    print(f"\n  Rates CSV → {CSV_RATES.name}")

    # -- Part 2: Regression --------------------------------------------------
    print("\n[3/5] Logit and LPM regressions ...")
    print("  Outcome: adopted_qk | Covariates: log(rbo), log(n_firms),")
    print("  a_net_influence, b_net_influence, a_entry_q_norm")

    reg_rows = []
    for k in HORIZONS:
        print(f"\n  --- Horizon Q+{k} ---")
        lo_full, lpm_full, n_full = run_logit(cand, k, "Full")
        print_reg_table(lo_full, lpm_full, n_full, "Full sample", k)
        reg_rows.append(extract_reg_row(lo_full, lpm_full, n_full, "full", k))

        lo_uniq, lpm_uniq, n_uniq = run_logit(cand_unique, k, "Unique entry")
        print_reg_table(lo_uniq, lpm_uniq, n_uniq, "Unique-entry robustness", k)
        reg_rows.append(extract_reg_row(lo_uniq, lpm_uniq, n_uniq, "unique_entry", k))

    pd.DataFrame(reg_rows).to_csv(CSV_REG, index=False)
    print(f"\n  Regression results CSV → {CSV_REG.name}")

    # -- Part 3: Who are B's that adopt? ------------------------------------
    print("\n[4/5] Profile of adopters vs. non-adopters (Q+1, full sample) ...")
    obs1  = cand[cand["horizon_obs_q1"] == 1].dropna(subset=["b_net_influence"])
    adopt = obs1[obs1["adopted_q1"] == 1]
    nonad = obs1[obs1["adopted_q1"] == 0]

    from scipy.stats import mannwhitneyu
    stat, p_mwu = mannwhitneyu(
        adopt["rbo_weight"], nonad["rbo_weight"], alternative="greater"
    )
    print(f"\n  Adopters (Q+1):     N={len(adopt):,}, "
          f"mean RBO={adopt['rbo_weight'].mean():.4f}, "
          f"mean B net_inf={adopt['b_net_influence'].mean():.2f}")
    print(f"  Non-adopters (Q+1): N={len(nonad):,}, "
          f"mean RBO={nonad['rbo_weight'].mean():.4f}, "
          f"mean B net_inf={nonad['b_net_influence'].mean():.2f}")
    print(f"  MWU rbo_weight (adopters > non-adopters): U={stat:.0f}, p={p_mwu:.4f}")

    # -- Part 4: A influence quartile breakdown at Q+1 ----------------------
    print("\n[5/5] Q+1 adoption rate by A's net_influence quartile ...")
    obs1_ni = cand[cand["horizon_obs_q1"] == 1].dropna(subset=["a_net_influence"]).copy()
    obs1_ni["a_ni_quartile"] = pd.qcut(
        obs1_ni["a_net_influence"], q=4,
        labels=["Q1 (weak)", "Q2", "Q3", "Q4 (strong)"]
    )
    print(f"\n  {'A influence quartile':<22} {'N':>8} {'Adopted':>9} {'Rate':>8}")
    print(f"  {'-'*52}")
    for qg in ["Q1 (weak)", "Q2", "Q3", "Q4 (strong)"]:
        sub = obs1_ni[obs1_ni["a_ni_quartile"] == qg]
        n_ad = sub["adopted_q1"].sum()
        r    = n_ad / len(sub) if len(sub) > 0 else np.nan
        print(f"  {qg:<22} {len(sub):>8,} {int(n_ad):>9} {r:>8.4f}")

    # -- Figure --------------------------------------------------------------
    make_adoption_curve(
        rates_full, rates_unique,
        FIG_DIR / "19_bill_adoption_diffusion.png",
    )

    print(f"\n  Candidate set CSV → {CSV_CAND}")
    print(f"  Adoption rates CSV → {CSV_RATES}")
    print(f"  Regression CSV    → {CSV_REG}")
    print(f"  Log               → {TXT_PATH}")
    print("\n  Validation complete.")
    print("=" * 72)

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
