"""
Payoff complementarity test — BCZ micro-level evidence (116th Congress).

Tests: if firm j newly enters bill b at quarter t, does firm i increase its
lobbying spend on bill b in t+1, and is that response larger when the (i, j)
RBO edge weight is high?

A positive and significant interaction coefficient (entry_j × rbo_ij) provides
micro-level evidence of strategic complementarity in the BCZ sense: best-response
spending increases in partner spending, and more so for structurally similar firms.

Regression specification:
    Δlog_spend_{i,b,t+1} = β₁ entry_{j,b,t} + β₂ rbo_ij
                          + β₃ (entry_{j,b,t} × rbo_ij)
                          + α_{i,b} + γ_t + ε

where:
  - Δlog_spend_{i,b,t+1} = log(spend_{i,b,t+1}) − log(spend_{i,b,t})
  - entry_{j,b,t}         = 1 if quarter t is the first quarter j lobbies bill b
  - rbo_ij                = RBO edge weight (congress-aggregate; symmetric)
  - α_{i,b}               = firm-bill fixed effects (within-pair demean)
  - γ_t                   = quarter fixed effects

Sample: RBO-linked pairs (i,j) only. Observations: one row per (i, j, b, t)
where firm i is active on bill b in quarters t and t+1 with positive spend,
and firm j first enters bill b at quarter t.

Three specifications:
  (A) Full RBO-linked sample
  (B) High-RBO pairs only (weight ≥ 75th percentile ≈ 0.095)
  (C) Low-RBO pairs only  (weight < 25th percentile ≈ 0.005)
  (D) Robustness: include all pairs (RBO-linked + non-linked), setting rbo=0
      for non-linked; tests whether entry effect exists even without RBO link

HC3 heteroskedasticity-robust standard errors throughout.
Standard errors clustered at the (firm_i, bill) level as secondary robustness.

Run from src/ directory:
  python validations/18_payoff_complementarity.py
"""

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

OUT_DIR  = ROOT / "outputs" / "validation"
TXT_PATH = OUT_DIR / "18_payoff_complementarity.txt"
CSV_PATH = OUT_DIR / "18_payoff_complementarity_panel.csv"
RES_PATH = OUT_DIR / "18_payoff_complementarity_results.csv"

# RBO weight quartile cutoffs (set after data load; placeholders here)
HIGH_RBO_PCT = 75   # top-quartile threshold for Spec B
LOW_RBO_PCT  = 25   # bottom-quartile threshold for Spec C


# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

class _Tee:
    """Write to stdout and a file simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------

def assign_quarters(df):
    """Add 'quarter' column (1–8): 2019 Q1-4 → 1-4, 2020 Q1-4 → 5-8."""
    df = df.copy()
    base_q   = df["report_type"].str[1].astype(int)
    year_off = df["year"].map({2019: 0, 2020: 4})
    df["quarter"] = base_q + year_off
    return df


def build_spend_panel(df_raw):
    """
    Aggregate to (firm, bill, quarter) → total amount_allocated.
    Returns DataFrame with columns: firm, bill, quarter, spend.
    """
    panel = (
        df_raw.groupby(["fortune_name", "bill_number", "quarter"])["amount_allocated"]
        .sum()
        .reset_index()
        .rename(columns={
            "fortune_name":    "firm",
            "bill_number":     "bill",
            "amount_allocated": "spend",
        })
    )
    return panel


def tag_entry_events(panel):
    """
    Add entry_j column to panel: 1 if this is the firm's first quarter on this
    bill, 0 otherwise. Returns the full panel with entry_j tagged.
    """
    first_q = panel.groupby(["firm", "bill"])["quarter"].min().reset_index().rename(
        columns={"quarter": "first_quarter"}
    )
    panel = panel.merge(first_q, on=["firm", "bill"])
    panel["entry_j"] = (panel["quarter"] == panel["first_quarter"]).astype(int)
    return panel.drop(columns=["first_quarter"])


def build_delta_log_spend(panel):
    """
    Build Δlog_spend for firm i on bill b from quarter t to t+1.
    Only consecutive quarters (t+1 = t+1) with positive spend in both periods.
    Returns DataFrame with columns: firm_i, bill, quarter (= t), delta_log_spend.
    """
    srt = panel.sort_values(["firm", "bill", "quarter"]).copy()
    srt["spend_next"]   = srt.groupby(["firm", "bill"])["spend"].shift(-1)
    srt["quarter_next"] = srt.groupby(["firm", "bill"])["quarter"].shift(-1)

    consec = srt[
        (srt["quarter_next"] == srt["quarter"] + 1) &
        (srt["spend"]        > 0) &
        (srt["spend_next"]   > 0)
    ].copy()
    consec["delta_log_spend"] = np.log(consec["spend_next"]) - np.log(consec["spend"])

    return consec[["firm", "bill", "quarter", "delta_log_spend"]].rename(
        columns={"firm": "firm_i"}
    )


def build_rbo_lookup(rbo_df):
    """Build symmetric RBO weight lookup: both (i→j) and (j→i) directions."""
    rbo_sym = pd.concat([
        rbo_df[["source", "target", "weight"]].rename(
            columns={"source": "firm_i", "target": "firm_j"}),
        rbo_df[["source", "target", "weight"]].rename(
            columns={"target": "firm_i", "source": "firm_j"}),
    ], ignore_index=True)
    return rbo_sym


def build_panel(delta_df, panel_with_entry, rbo_sym, include_non_linked=False):
    """
    Construct the regression panel by joining:
      - firm_i's Δlog_spend at quarter t (from consecutive positive-spend transitions)
      - ALL firm_j active on the same bill at quarter t, with entry_j flag
        (entry_j=1 if first quarter j lobbies bill b, 0 if continuing)
      - RBO weight for the (i, j) pair

    Crucially, firm_j rows include BOTH entering and continuing lobbyists,
    which gives entry_j real within-group variation needed for identification.

    include_non_linked: if True, keep rows where (i,j) have no RBO edge
                        (rbo_ij = 0); used for Spec D robustness.

    Returns DataFrame ready for regression.
    """
    # firm_j status table: (firm_j, bill, quarter, entry_j)
    j_status = panel_with_entry[["firm", "bill", "quarter", "entry_j"]].rename(
        columns={"firm": "firm_j"}
    )

    # Cross firm_i's Δlog_spend with all active firm_j on same (bill, quarter)
    merged = delta_df.merge(j_status, on=["bill", "quarter"], how="inner")
    merged = merged[merged["firm_i"] != merged["firm_j"]]   # drop self-pairs

    # Join RBO weights
    merged = merged.merge(rbo_sym, on=["firm_i", "firm_j"], how="left")

    if include_non_linked:
        merged["rbo_ij"] = merged["weight"].fillna(0.0)
        merged["has_rbo_edge"] = merged["weight"].notna().astype(int)
    else:
        merged = merged[merged["weight"].notna()].copy()
        merged["rbo_ij"] = merged["weight"]

    merged = merged.drop(columns=["weight"])

    # Interaction term
    merged["entry_x_rbo"] = merged["entry_j"] * merged["rbo_ij"]

    # Firm-bill FE identifier
    merged["firm_bill"] = merged["firm_i"] + "||" + merged["bill"]

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# OLS with firm-bill and quarter fixed effects (within-transformation)
# ---------------------------------------------------------------------------

def demean_within(df, group_col, value_cols):
    """Subtract group mean for each value column (within-group demeaning)."""
    df = df.copy()
    gm = df.groupby(group_col)[value_cols].transform("mean")
    for c in value_cols:
        df[c + "_dm"] = df[c] - gm[c]
    return df


def run_ols_spec(df, spec_label, cluster_col=None):
    """
    Run within-transformed OLS with quarter FE and HC3 SE.
    Within-transformation absorbs firm-bill FE.
    Quarter FE added as dummies after within-transformation.
    Returns statsmodels RegressionResultsWrapper.
    """
    # Within-demean to absorb firm-bill FE
    cols_to_demean = ["delta_log_spend", "entry_j", "rbo_ij", "entry_x_rbo"]
    df_dm = demean_within(df, "firm_bill", cols_to_demean)

    # Quarter dummies (within-transformed too)
    quarter_dummies = pd.get_dummies(df["quarter"], prefix="q", drop_first=True)
    for col in quarter_dummies.columns:
        quarter_dummies[col] = quarter_dummies[col].astype(float)
        gm = quarter_dummies[col].groupby(df["firm_bill"]).transform("mean")
        df_dm[col + "_dm"] = quarter_dummies[col] - gm

    q_dm_cols = [c for c in df_dm.columns if c.endswith("_dm") and c.startswith("q")]

    # Regressor matrix
    X_cols = ["entry_j_dm", "rbo_ij_dm", "entry_x_rbo_dm"] + q_dm_cols
    X = df_dm[X_cols].copy()
    X.insert(0, "const", 0.0)   # constant absorbs to 0 after within-transform
    y = df_dm["delta_log_spend_dm"]

    # Drop singletons (firm-bill groups with only 1 obs — no within variation)
    mask = df.groupby("firm_bill")["firm_bill"].transform("count") > 1
    X = X[mask]
    y = y[mask]

    result = sm.OLS(y, X).fit(cov_type="HC3")
    return result, mask.sum()


def print_ols_table(result, n_obs, spec_label):
    """Print compact OLS results table for the key regressors."""
    print(f"\n  Spec {spec_label}  (N={n_obs:,})")
    print(f"  {'Variable':<22} {'Coef':>10} {'SE(HC3)':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*60}")

    key_vars = ["entry_j_dm", "rbo_ij_dm", "entry_x_rbo_dm"]
    labels   = {"entry_j_dm": "entry_j", "rbo_ij_dm": "rbo_ij",
                "entry_x_rbo_dm": "entry_j × rbo_ij"}
    for var in key_vars:
        if var not in result.params.index:
            continue
        b  = result.params[var]
        se = result.bse[var]
        t  = result.tvalues[var]
        p  = result.pvalues[var]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {labels[var]:<22} {b:>10.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f} {stars}")

    print(f"  {'R² (within)':.<22} {result.rsquared:>10.4f}")
    print(f"  {'Adj. R²':.<22} {result.rsquared_adj:>10.4f}")


def extract_results_row(result, n_obs, spec_label):
    """Extract key results as a dict for CSV export."""
    row = {"spec": spec_label, "n_obs": n_obs,
           "r2": round(result.rsquared, 5),
           "adj_r2": round(result.rsquared_adj, 5)}
    for var, label in [("entry_j_dm", "entry_j"),
                       ("rbo_ij_dm", "rbo_ij"),
                       ("entry_x_rbo_dm", "entry_x_rbo")]:
        if var in result.params.index:
            row[f"coef_{label}"]  = round(result.params[var], 5)
            row[f"se_{label}"]    = round(result.bse[var], 5)
            row[f"t_{label}"]     = round(result.tvalues[var], 4)
            row[f"p_{label}"]     = round(result.pvalues[var], 5)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 72)
    print("VALIDATION 18: PAYOFF COMPLEMENTARITY TEST (BCZ, 116th CONGRESS)")
    print("=" * 72)
    print()
    print("Specification:")
    print("  Δlog_spend_{i,b,t+1} = β₁ entry_{j,b,t} + β₂ rbo_ij")
    print("                        + β₃ (entry_{j,b,t} × rbo_ij)")
    print("                        + α_{i,b} + γ_t + ε")
    print()
    print("  α_{i,b} = firm-bill FE (within-transformation)")
    print("  γ_t     = quarter FE (dummies, within-transformed)")
    print("  SE      = HC3 heteroskedasticity-robust")
    print()

    # -- Load data -----------------------------------------------------------
    print("[1/4] Loading and constructing panel ...")
    df_raw = pd.read_csv(
        DATA_DIR / "congress" / "116" / "opensecrets_lda_reports.csv"
    )
    df_raw = assign_quarters(df_raw)

    rbo = pd.read_csv(
        DATA_DIR / "congress" / "116" / "rbo_directed_influence.csv"
    )

    panel           = build_spend_panel(df_raw)
    panel_with_entry = tag_entry_events(panel)
    delta_df        = build_delta_log_spend(panel)
    rbo_sym         = build_rbo_lookup(rbo)

    n_entries = (panel_with_entry["entry_j"] == 1).sum()
    print(f"  (firm, bill, quarter) spend triples: {len(panel):,}")
    print(f"  Entry events (firm_j first-time):    {n_entries:,}")
    print(f"  Δlog_spend observations:             {len(delta_df):,}")
    print(f"  Symmetric RBO pairs:                 {len(rbo_sym):,}")

    # -- Build main panel (RBO-linked pairs only) ----------------------------
    main_panel = build_panel(delta_df, panel_with_entry, rbo_sym, include_non_linked=False)
    print(f"\n  Main panel (RBO-linked, i≠j):        {len(main_panel):,} obs")
    print(f"  Unique firm_i:                       {main_panel['firm_i'].nunique()}")
    print(f"  Unique firm_j:                       {main_panel['firm_j'].nunique()}")
    print(f"  Unique (firm_i, bill) groups:        {main_panel['firm_bill'].nunique()}")
    print(f"  Quarters covered:                    {sorted(main_panel['quarter'].unique())}")

    # RBO weight cutoffs for Specs B and C
    rbo_q75 = np.percentile(main_panel["rbo_ij"], HIGH_RBO_PCT)
    rbo_q25 = np.percentile(main_panel["rbo_ij"], LOW_RBO_PCT)
    print(f"\n  RBO weight p25={rbo_q25:.4f}  p75={rbo_q75:.4f}")

    # Δlog_spend descriptive
    print(f"\n  Δlog_spend (main panel) describe:")
    desc = main_panel["delta_log_spend"].describe()
    for k, v in desc.items():
        print(f"    {k:<8} {v:>10.4f}")

    # Save panel CSV for inspection
    main_panel.to_csv(CSV_PATH, index=False)
    print(f"\n  Panel CSV → {CSV_PATH.name}")

    # -- Analysis 1: Summary by RBO quartile --------------------------------
    print("\n[2/4] Descriptive: mean Δlog_spend by RBO quartile and entry status ...")
    main_panel["rbo_quartile"] = pd.qcut(
        main_panel["rbo_ij"], q=4,
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )
    tbl = (
        main_panel.groupby("rbo_quartile", observed=True)["delta_log_spend"]
        .agg(n="count", mean="mean", std="std", median="median")
        .round(4)
    )
    print(f"\n  Mean Δlog_spend by RBO quartile (all entry events):")
    print(f"  {'Quartile':<14} {'N':>7} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print(f"  {'-'*50}")
    for q, row in tbl.iterrows():
        print(f"  {str(q):<14} {int(row['n']):>7} {row['mean']:>8.4f} "
              f"{row['std']:>8.4f} {row['median']:>8.4f}")

    # -- Regressions ---------------------------------------------------------
    print("\n[3/4] Running regressions ...")
    results_rows = []

    # Spec A: full RBO-linked sample
    print("\n  Spec A — Full RBO-linked panel")
    res_a, n_a = run_ols_spec(main_panel, "A")
    print_ols_table(res_a, n_a, "A: Full RBO-linked")
    results_rows.append(extract_results_row(res_a, n_a, "A_full_rbo_linked"))

    # Spec B: high-RBO pairs (≥ p75)
    high_rbo = main_panel[main_panel["rbo_ij"] >= rbo_q75].copy()
    print(f"\n  Spec B — High-RBO pairs (rbo ≥ {rbo_q75:.4f}; N={len(high_rbo):,})")
    res_b, n_b = run_ols_spec(high_rbo, "B")
    print_ols_table(res_b, n_b, "B: High-RBO (≥p75)")
    results_rows.append(extract_results_row(res_b, n_b, "B_high_rbo_p75"))

    # Spec C: low-RBO pairs (< p25)
    low_rbo = main_panel[main_panel["rbo_ij"] < rbo_q25].copy()
    print(f"\n  Spec C — Low-RBO pairs (rbo < {rbo_q25:.4f}; N={len(low_rbo):,})")
    res_c, n_c = run_ols_spec(low_rbo, "C")
    print_ols_table(res_c, n_c, "C: Low-RBO (<p25)")
    results_rows.append(extract_results_row(res_c, n_c, "C_low_rbo_p25"))

    # Spec D: all pairs (RBO-linked + non-linked, rbo=0 for non-linked)
    full_panel = build_panel(delta_df, panel_with_entry, rbo_sym, include_non_linked=True)
    print(f"\n  Spec D — All pairs (RBO-linked + non-linked, rbo=0 if no edge; "
          f"N={len(full_panel):,})")
    res_d, n_d = run_ols_spec(full_panel, "D")
    print_ols_table(res_d, n_d, "D: All pairs (rbo=0 if no edge)")
    results_rows.append(extract_results_row(res_d, n_d, "D_all_pairs"))

    # -- Summary table -------------------------------------------------------
    print("\n[4/4] Summary ...")
    print()
    print("  Key coefficient: β₃ = entry_j × rbo_ij (BCZ complementarity)")
    print()
    print(f"  {'Spec':<35} {'β₃ (coef)':>12} {'SE':>8} {'p':>8} {'Interp.'}")
    print(f"  {'-'*80}")
    for row in results_rows:
        b3  = row.get("coef_entry_x_rbo", np.nan)
        se  = row.get("se_entry_x_rbo",   np.nan)
        p   = row.get("p_entry_x_rbo",    np.nan)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(ns)"
        interp = (
            "Complementarity supported"  if b3 > 0 and p < 0.05 else
            "No complementarity"         if b3 <= 0 or p >= 0.05 else
            "Negative (substitutability)"
        )
        print(f"  {row['spec']:<35} {b3:>12.4f} {se:>8.4f} {p:>8.4f} {sig}  {interp}")

    print()
    print("  Interpretation notes:")
    print("  - β₁ (entry_j): baseline response to any co-lobbyist entry")
    print("  - β₂ (rbo_ij):  level difference in spend growth for high-RBO pairs")
    print("  - β₃ (interaction): marginal effect of entry for each unit of RBO weight")
    print("  - Positive β₃ + significance → BCZ payoff complementarity confirmed")
    print("  - Positive β₁ alone → herding without structural complementarity")

    # Save results CSV
    pd.DataFrame(results_rows).to_csv(RES_PATH, index=False)
    print(f"\n  Results CSV  → {RES_PATH}")
    print(f"  Panel CSV    → {CSV_PATH}")
    print(f"  Log          → {TXT_PATH}")
    print("\n  Validation complete.")
    print("=" * 72)

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
