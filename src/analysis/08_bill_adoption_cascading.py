"""
Bill adoption cascading — predicting lobbying actions through influence (116th).

Answers:
  1. Is a net follower B more likely to adopt a bill that influencer A lobbied,
     within Q+1, Q+2, and Q+3 after A's entry?
  2. Is that adoption rate higher for high-RBO pairs vs. low-RBO pairs?
  3. Regression: does rbo_weight predict adoption probability beyond controls?
  4. Case studies: specific influencer→follower→bill cascade chains.

Unit of analysis: (A, B, bill) exposure events where A enters bill b at
quarter t and B had not yet lobbied b at or before t.

Outputs (outputs/analysis/):
  08_adoption_rates.csv
  08_adoption_regression.csv
  08_cascade_case_studies.csv
  08_bill_adoption_cascading.txt
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CONGRESS = 116
HORIZONS = [1, 2, 3]
MAX_Q    = 8

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
# Data construction helpers
# ---------------------------------------------------------------------------

def assign_quarters(df):
    df = df.copy()
    base_q   = df["report_type"].str[1].astype(int)
    year_off = df["year"].map({2019: 0, 2020: 4})
    df["quarter"] = base_q + year_off
    return df

def build_first_quarter_table(df_raw):
    """(firm, bill, first_q) for positive-spend rows; also bill_firms count."""
    panel = (
        df_raw.groupby(["fortune_name", "bill_number", "quarter"])["amount_allocated"]
        .sum().reset_index()
        .rename(columns={"fortune_name": "firm", "bill_number": "bill",
                         "amount_allocated": "spend"})
    )
    panel = panel[panel["spend"] > 0]
    first_q  = (panel.groupby(["firm", "bill"])["quarter"].min()
                .reset_index().rename(columns={"quarter": "first_q"}))
    bill_firms = panel.groupby("bill")["firm"].nunique().rename("n_firms_on_bill")
    return first_q, bill_firms

def build_candidate_set(first_q_df, directed_df, bill_firms, node_attrs):
    """One row per (A, B, bill) where A enters bill b before B."""
    a_first = first_q_df.rename(columns={"firm": "A", "first_q": "a_entry_q"})
    cand    = directed_df.merge(a_first, on="A")

    b_first = first_q_df.rename(columns={"firm": "B", "first_q": "b_first_q"})
    cand    = cand.merge(b_first, on=["B", "bill"], how="left")

    b_not_yet = cand["b_first_q"].isna() | (cand["b_first_q"] > cand["a_entry_q"])
    cand = cand[b_not_yet].copy()

    for k in HORIZONS:
        cand[f"adopted_q{k}"] = (
            cand["b_first_q"].notna() &
            (cand["b_first_q"] <= cand["a_entry_q"] + k)
        ).astype(int)
        cand[f"horizon_obs_q{k}"] = (cand["a_entry_q"] + k <= MAX_Q).astype(int)

    cand = cand.merge(bill_firms, on="bill", how="left")

    a_attr = (node_attrs[["firm", "net_influence"]]
              .rename(columns={"firm": "A", "net_influence": "a_net_influence"}))
    b_attr = (node_attrs[["firm", "net_influence"]]
              .rename(columns={"firm": "B", "net_influence": "b_net_influence"}))
    cand = cand.merge(a_attr, on="A", how="left").merge(b_attr, on="B", how="left")

    cand["log_rbo"]       = np.log(cand["rbo_weight"] + 1e-8)
    cand["log_n_firms"]   = np.log(cand["n_firms_on_bill"].clip(lower=1))
    cand["a_entry_q_norm"] = cand["a_entry_q"] / MAX_Q
    return cand.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Adoption rate table
# ---------------------------------------------------------------------------

def adoption_rate_table(cand):
    """Adoption rates at Q+1/2/3 by RBO quartile."""
    cand = cand.copy()
    cand["rbo_quartile"] = pd.qcut(cand["rbo_weight"], q=4,
                                   labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    rows = []
    for k in HORIZONS:
        obs_all = cand[cand[f"horizon_obs_q{k}"] == 1]
        r_all   = obs_all[f"adopted_q{k}"].mean()
        rows.append({"horizon": f"Q+{k}", "rbo_group": "All",
                     "n": len(obs_all), "rate": round(r_all, 5)})
        for qg in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
            sub = obs_all[obs_all["rbo_quartile"] == qg]
            rows.append({"horizon": f"Q+{k}", "rbo_group": qg,
                         "n": len(sub), "rate": round(sub[f"adopted_q{k}"].mean(), 5)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# LPM regression
# ---------------------------------------------------------------------------

def run_lpm(cand, k):
    """Linear probability model for P(adopted within Q+k)."""
    obs = cand[cand[f"horizon_obs_q{k}"] == 1].copy()
    covars = ["log_rbo", "log_n_firms", "a_net_influence",
              "b_net_influence", "a_entry_q_norm"]
    obs = obs.dropna(subset=[f"adopted_q{k}"] + covars)
    y   = obs[f"adopted_q{k}"].astype(float)
    X   = sm.add_constant(obs[covars].astype(float))
    res = sm.OLS(y, X).fit(cov_type="HC3")
    return res, len(obs)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT_DIR / "08_bill_adoption_cascading.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    SEP = "=" * 70
    print(SEP)
    print("ANALYSIS 08: BILL ADOPTION CASCADING — 116th CONGRESS")
    print(SEP)
    print()
    print("Question: Is a follower B more likely to adopt a bill that influencer")
    print("A lobbied, and does higher RBO amplify that adoption probability?")

    # -- Load data -------------------------------------------------------
    print("\n[1/4] Building candidate set ...")
    df_raw     = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/opensecrets_lda_reports.csv")
    df_raw     = assign_quarters(df_raw)
    rbo        = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/rbo_directed_influence.csv")
    node_attrs = pd.read_csv(DATA_DIR / f"congress/{CONGRESS}/node_attributes.csv")

    # Decisive directed pairs (A leads B)
    directed = (
        rbo[rbo["net_temporal"] > 0][["source", "target", "rbo"]]
        .rename(columns={"source": "A", "target": "B", "rbo": "rbo_weight"})
        .copy()
    )
    print(f"  Directed (A→B) pairs:  {len(directed):,}")

    first_q_df, bill_firms = build_first_quarter_table(df_raw)
    cand = build_candidate_set(first_q_df, directed, bill_firms, node_attrs)

    print(f"  Candidate (A, B, bill) rows:  {len(cand):,}")
    print(f"  A firms:                      {cand['A'].nunique()}")
    print(f"  B firms:                      {cand['B'].nunique()}")
    for k in HORIZONS:
        n_obs = cand[f"horizon_obs_q{k}"].sum()
        print(f"  Observable at Q+{k}: {n_obs:,}")

    # -- Adoption rates --------------------------------------------------
    print("\n[2/4] Adoption rates by RBO quartile and horizon ...")
    rates = adoption_rate_table(cand)
    rates.to_csv(OUT_DIR / "08_adoption_rates.csv", index=False)

    rbo_med = cand["rbo_weight"].median()
    print(f"\n  RBO median: {rbo_med:.4f}")
    print(f"\n  {'Horizon':<8} {'Group':<14} {'N':>8} {'Rate':>8}")
    print(f"  {'─'*44}")
    for _, row in rates.iterrows():
        print(f"  {row['horizon']:<8} {row['rbo_group']:<14} "
              f"{int(row['n']):>8,} {row['rate']:>8.4f}")

    # Median-split summary
    print(f"\n  Median-split adoption rates:")
    print(f"  {'Horizon':<8} {'High RBO':>12} {'Low RBO':>12} {'Ratio':>8}")
    print(f"  {'─'*44}")
    for k in HORIZONS:
        obs = cand[cand[f"horizon_obs_q{k}"] == 1]
        hi  = obs[obs["rbo_weight"] >= rbo_med][f"adopted_q{k}"].mean()
        lo  = obs[obs["rbo_weight"] <  rbo_med][f"adopted_q{k}"].mean()
        ratio = hi / lo if lo > 0 else float("nan")
        print(f"  Q+{k}      {hi:>12.4f} {lo:>12.4f} {ratio:>8.3f}x")

    # Chi-squared test at Q+1
    obs1 = cand[cand["horizon_obs_q1"] == 1]
    hi1  = obs1[obs1["rbo_weight"] >= rbo_med]
    lo1  = obs1[obs1["rbo_weight"] <  rbo_med]
    ct = np.array([
        [hi1["adopted_q1"].sum(), len(hi1) - hi1["adopted_q1"].sum()],
        [lo1["adopted_q1"].sum(), len(lo1) - lo1["adopted_q1"].sum()],
    ])
    chi2, p_chi, _, _ = chi2_contingency(ct)
    print(f"\n  χ² test (high vs low RBO, Q+1): χ²={chi2:.3f}, p={p_chi:.4f}")

    # -- Regression ------------------------------------------------------
    print("\n[3/4] LPM regressions ...")
    print(f"\n  {'Horizon':<8} {'N':>8} {'coef_log_rbo':>14} {'p':>10}")
    print(f"  {'─'*44}")
    reg_rows = []
    for k in HORIZONS:
        res, n = run_lpm(cand, k)
        b   = res.params.get("log_rbo", np.nan)
        p   = res.pvalues.get("log_rbo", np.nan)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Q+{k}      {n:>8,} {float(b):>14.5f} {float(p):>10.4f} {stars}")
        reg_rows.append({
            "horizon": f"Q+{k}", "n": n,
            "coef_log_rbo": round(float(b), 6),
            "se_log_rbo": round(float(res.bse.get("log_rbo", np.nan)), 6),
            "p_log_rbo":  round(float(p), 6),
            "r2":          round(res.rsquared, 5),
        })
    pd.DataFrame(reg_rows).to_csv(OUT_DIR / "08_adoption_regression.csv", index=False)

    # -- Case studies ----------------------------------------------------
    print("\n[4/4] Case study cascade chains ...")
    # For each top influencer (by net_influence), show their top-adopted followers
    top_A = node_attrs.nlargest(10, "net_influence")["firm"].tolist()
    case_rows = []
    print(f"\n  Top influencer cascades (A→B, adopted within Q+1):")
    print(f"  {'A (influencer)':<38} {'B (follower)':<38} "
          f"{'rbo':>8} {'bill':<18} {'q_A':>5} {'q_B':>5}")
    print(f"  {'─'*115}")

    obs1 = cand[
        (cand["A"].isin(top_A)) &
        (cand["horizon_obs_q1"] == 1) &
        (cand["adopted_q1"] == 1)
    ].sort_values(["A", "rbo_weight"], ascending=[True, False])

    shown = 0
    for _, row in obs1.iterrows():
        if shown >= 25:
            break
        print(f"  {row['A']:<38} {row['B']:<38} "
              f"{row['rbo_weight']:>8.4f} {row['bill']:<18} "
              f"{int(row['a_entry_q']):>5} {int(row['b_first_q']):>5}")
        case_rows.append({
            "influencer": row["A"], "follower": row["B"],
            "bill": row["bill"], "rbo_weight": row["rbo_weight"],
            "a_entry_q": int(row["a_entry_q"]), "b_entry_q": int(row["b_first_q"]),
            "lag": int(row["b_first_q"]) - int(row["a_entry_q"]),
        })
        shown += 1

    pd.DataFrame(case_rows).to_csv(OUT_DIR / "08_cascade_case_studies.csv", index=False)

    print(f"\n  Outputs:")
    print(f"    08_adoption_rates.csv")
    print(f"    08_adoption_regression.csv")
    print(f"    08_cascade_case_studies.csv")
    print(f"    08_bill_adoption_cascading.txt")
    print(f"\n{SEP}")
    print("Analysis complete.")

    log_f.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
