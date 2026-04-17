"""
Regression analysis: predictors of firm influencer status (116th and 117th Congress).

Three OLS regressions per congress (six total):
  (A) net_influence ~ log_spend + log_bills + katz_centrality + participation_coeff
  (B) net_strength  ~ log_spend + log_bills + katz_centrality + participation_coeff
  (C) wc_net_strength ~ log_spend + log_bills + wc_eigenvector + wc_pagerank
      (within-community measures; 116th community partition applied to both congresses;
       firms without a 116th community label are dropped for regression C only)

Also reports OLS with a top-quartile binary outcome (indicator for net_influence ≥ 75th
percentile) for interpretability alongside the continuous-outcome regressions.

Covariates sourced from:
  - data/congress/{num}/opensecrets_lda_reports.csv  → total_spend, num_bills
  - data/centralities/centrality_affiliation.csv      → katz_centrality, participation_coeff,
                                                         within_comm_eigenvector
  - validation 13 output                              → wc_pagerank (recomputed here)
  - data/communities/communities_affiliation.csv      → community partition for wc_net_strength
  - data/congress/{num}/node_attributes.csv           → net_influence, net_strength (outcomes)
  - data/congress/{num}/rbo_directed_influence.csv    → wc_net_strength computation

Centrality covariates (katz, participation_coeff, wc_eigenvector) are taken from the
116th-Congress affiliation network for both congresses. This is a deliberate simplification:
per-congress centrality recomputation would require re-running bill_affiliation_network.py
for the 117th, which is out of scope here. The 116th affiliation network is used as the
structural baseline. Firms present in the 117th but absent from the 116th centrality table
(42 firms) are dropped from regressions A, B, and C for the 117th.

Run from src/ directory:
  python validations/14_influencer_regression.py
"""

import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr

import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, ROOT
from utils.centrality import compute_katz_centrality

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------

OUT_DIR  = ROOT / "outputs" / "validation"
CSV_PATH = OUT_DIR / "14_influencer_regression.csv"
TXT_PATH = OUT_DIR / "14_influencer_regression.txt"

CONGRESSES     = [116, 117]
WEIGHT_COL     = "weight"
MIN_SPEND      = 1.0   # floor for log transform (dollars; avoids log(0))
MIN_BILLS      = 1     # floor for log transform


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
# Covariate builders
# ---------------------------------------------------------------------------

def build_spend_bills(congress):
    """Total spend and unique bill count per firm from opensecrets_lda_reports.csv."""
    rpt = pd.read_csv(DATA_DIR / f"congress/{congress}/opensecrets_lda_reports.csv")
    # amount_allocated is per-report (not per bill-row); deduplicate on uniq_id x firm
    uniq = rpt.drop_duplicates(subset=["uniq_id", "fortune_name"])
    spend = uniq.groupby("fortune_name")["amount_allocated"].sum().rename("total_spend")
    bills = rpt.groupby("fortune_name")["bill_number"].nunique().rename("num_bills")
    df = pd.concat([spend, bills], axis=1).reset_index().rename(columns={"fortune_name": "firm"})
    df["log_spend"] = np.log(df["total_spend"].clip(lower=MIN_SPEND))
    df["log_bills"] = np.log(df["num_bills"].clip(lower=MIN_BILLS))
    return df[["firm", "total_spend", "num_bills", "log_spend", "log_bills"]]


def load_centrality_116():
    """
    Load 116th-Congress affiliation centrality table.
    Used as structural baseline for both 116th and 117th regressions.
    """
    cent = pd.read_csv(DATA_DIR / "centralities" / "centrality_affiliation.csv")
    return cent[["firm", "katz_centrality", "participation_coeff",
                 "within_comm_eigenvector"]]


def compute_wc_pagerank_116():
    """
    Within-community PageRank on the 116th affiliation graph using stored community labels.
    Mirrors the computation in validation 13.
    """
    edges = pd.read_csv(DATA_DIR / "network_edges" / "affiliation_edges.csv")
    comm  = pd.read_csv(DATA_DIR / "communities" / "communities_affiliation.csv")
    G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr=WEIGHT_COL)
    partition = dict(zip(comm["fortune_name"], comm["community_aff"]))

    comm_map = {}
    for node, cid in partition.items():
        comm_map.setdefault(cid, []).append(node)

    pr = {}
    for cid, members in comm_map.items():
        sub = G.subgraph(members)
        if sub.number_of_nodes() < 2:
            pr.update({n: 0.0 for n in members})
        else:
            pr.update(nx.pagerank(sub, weight=WEIGHT_COL))

    return pd.Series(pr, name="wc_pagerank").rename_axis("firm").reset_index()


def build_wc_net_strength(congress, partition):
    """
    Within-community net_strength for a given congress.
    Uses the stored 116th-Congress affiliation community partition.
    Firms without a partition label are returned as NaN.
    """
    directed = pd.read_csv(DATA_DIR / f"congress/{congress}/rbo_directed_influence.csv")
    directed = directed[directed["balanced"] == 0].copy()

    nodes = pd.read_csv(DATA_DIR / f"congress/{congress}/node_attributes.csv")
    firms = nodes["firm"].tolist()

    def comm(f):
        return partition.get(f)

    results = {}
    for firm in firms:
        my_comm = comm(firm)
        if my_comm is None:
            results[firm] = np.nan
            continue
        as_src = directed[
            (directed["source"] == firm) &
            (directed["target"].map(comm) == my_comm)
        ]
        as_tgt = directed[
            (directed["target"] == firm) &
            (directed["source"].map(comm) == my_comm)
        ]
        results[firm] = as_src["weight"].sum() - as_tgt["weight"].sum()

    return (pd.Series(results, name="wc_net_strength")
              .rename_axis("firm").reset_index())


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def _top_quartile_indicator(series):
    """Binary indicator: 1 if value >= 75th percentile, else 0."""
    threshold = series.quantile(0.75)
    return (series >= threshold).astype(int)


def run_ols(df, formula, label, top_q_outcome=None):
    """
    Run OLS regression and return a results summary dict.
    If top_q_outcome is provided, also runs a logit for the binary indicator.
    Returns (ols_result, logit_result_or_None, summary_rows).
    """
    model = smf.ols(formula=formula, data=df.dropna()).fit(
        cov_type="HC3"  # heteroskedasticity-robust standard errors
    )
    return model


def print_ols_table(result, label):
    """Print compact OLS table: coefficients, robust SEs, t-stats, p-values, R²."""
    print(f"\n  {label}")
    print(f"  N={int(result.nobs)}, R²={result.rsquared:.4f}, adj-R²={result.rsquared_adj:.4f}, "
          f"F(HC3-adj) p={result.f_pvalue:.4f}")
    print(f"  {'Variable':<35} {'Coef':>10} {'Robust SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*71}")
    for varname in result.model.exog_names:
        coef   = result.params[varname]
        se     = result.bse[varname]
        tstat  = result.tvalues[varname]
        pval   = result.pvalues[varname]
        stars  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {varname:<35} {coef:>10.4f} {se:>10.4f} {tstat:>8.3f} {pval:>8.4f} {stars}")
    print(f"  {'-'*71}")


def ols_to_row(result, congress, outcome, spec):
    """Convert OLS result to a flat dict for the output CSV."""
    row = {
        "congress":  congress,
        "outcome":   outcome,
        "spec":      spec,
        "n":         int(result.nobs),
        "r2":        round(result.rsquared, 4),
        "adj_r2":    round(result.rsquared_adj, 4),
        "f_pvalue":  round(result.f_pvalue, 5),
    }
    for varname in result.model.exog_names:
        safe = varname.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
        row[f"coef_{safe}"]  = round(result.params[varname], 5)
        row[f"se_{safe}"]    = round(result.bse[varname], 5)
        row[f"pval_{safe}"]  = round(result.pvalues[varname], 5)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(TXT_PATH, "w")
    sys.stdout = _Tee(sys.__stdout__, log_f)

    print("=" * 75)
    print("VALIDATION 14: INFLUENCER REGRESSION (116th AND 117th CONGRESS)")
    print("=" * 75)
    print()
    print("Outcomes:  net_influence, net_strength, wc_net_strength")
    print("Also runs: top-quartile OLS for net_influence (binary indicator)")
    print("Covariates: log_spend, log_bills, katz_centrality,")
    print("            participation_coeff, within_comm_eigenvector, wc_pagerank")
    print("Note: centrality covariates are from the 116th-Congress affiliation")
    print("      network (structural baseline for both congresses).")
    print("SE: HC3 heteroskedasticity-robust throughout.")

    # -- Load shared covariates -----------------------------------------------

    print("\n[1/4] Loading shared covariates ...")
    cent116      = load_centrality_116()
    print(f"      Centrality (116th affiliation): {len(cent116)} firms")

    print("      Computing within-community PageRank (116th affiliation) ...")
    wc_pr_df     = compute_wc_pagerank_116()
    print(f"      WC PageRank computed: {len(wc_pr_df)} firms")

    comm_df      = pd.read_csv(DATA_DIR / "communities" / "communities_affiliation.csv")
    partition    = dict(zip(comm_df["fortune_name"], comm_df["community_aff"]))

    # Merge centrality covariates into one table (indexed on firm)
    cent_merged = cent116.merge(wc_pr_df, on="firm", how="left")

    # -- Run regressions per congress -----------------------------------------

    all_rows = []

    for congress in CONGRESSES:
        print(f"\n[2–3/4] Congress: {congress}th")

        # Outcomes
        nodes = pd.read_csv(DATA_DIR / f"congress/{congress}/node_attributes.csv")

        # Spend/bills covariates
        spend_df = build_spend_bills(congress)

        # Within-community net_strength
        wc_ns_df = build_wc_net_strength(congress, partition)

        # Build analysis dataset
        df = (nodes
              .merge(spend_df,    on="firm", how="left")
              .merge(cent_merged, on="firm", how="left")
              .merge(wc_ns_df,    on="firm", how="left"))

        # Top-quartile indicator for net_influence
        df["top_q_ni"] = _top_quartile_indicator(df["net_influence"])

        print(f"  Merged dataset: {len(df)} firms")
        print(f"  Complete cases (all regressors): "
              f"{df[['net_influence','net_strength','log_spend','log_bills','katz_centrality','participation_coeff']].dropna().shape[0]}")

        print(f"\n  --- {congress}th Congress ---")

        # Spec A: net_influence ~ log_spend + log_bills + katz + P
        formulaA = ("net_influence ~ log_spend + log_bills "
                    "+ katz_centrality + participation_coeff")
        resA = run_ols(df, formulaA, f"(A) net_influence ~ log_spend + log_bills + katz + P")
        print_ols_table(resA, f"(A) net_influence [{congress}th]")
        all_rows.append(ols_to_row(resA, congress, "net_influence", "A"))

        # Spec A2: top-quartile indicator (OLS linear probability)
        formulaA2 = ("top_q_ni ~ log_spend + log_bills "
                     "+ katz_centrality + participation_coeff")
        resA2 = run_ols(df, formulaA2, f"(A2) top_quartile_ni ~ log_spend + log_bills + katz + P")
        print_ols_table(resA2, f"(A2) top_quartile net_influence [{congress}th]")
        all_rows.append(ols_to_row(resA2, congress, "top_quartile_net_influence", "A2"))

        # Spec B: net_strength ~ log_spend + log_bills + katz + P
        formulaB = ("net_strength ~ log_spend + log_bills "
                    "+ katz_centrality + participation_coeff")
        resB = run_ols(df, formulaB, f"(B) net_strength ~ log_spend + log_bills + katz + P")
        print_ols_table(resB, f"(B) net_strength [{congress}th]")
        all_rows.append(ols_to_row(resB, congress, "net_strength", "B"))

        # Spec C: wc_net_strength ~ log_spend + log_bills + wc_eigenvector + wc_pagerank
        # Drop firms with no community label (117th firms absent from 116th partition)
        df_c = df.dropna(subset=["wc_net_strength", "wc_pagerank", "within_comm_eigenvector"])
        print(f"\n  Spec C (within-community): {len(df_c)} firms with community labels")
        formulaC = ("wc_net_strength ~ log_spend + log_bills "
                    "+ within_comm_eigenvector + wc_pagerank")
        resC = run_ols(df_c, formulaC, f"(C) wc_net_strength ~ log_spend + log_bills + wc_eigen + wc_pr")
        print_ols_table(resC, f"(C) wc_net_strength [{congress}th]")
        all_rows.append(ols_to_row(resC, congress, "wc_net_strength", "C"))

        # Descriptive summary of outcome variables
        print(f"\n  Outcome descriptives [{congress}th]:")
        for col in ["net_influence", "net_strength", "wc_net_strength"]:
            s = df[col].dropna()
            q75 = s.quantile(0.75)
            print(f"    {col:<22}: n={len(s)}, mean={s.mean():.3f}, "
                  f"sd={s.std():.3f}, p25={s.quantile(0.25):.3f}, "
                  f"p75={q75:.3f}")

    # -- Save CSV -------------------------------------------------------------

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(CSV_PATH, index=False)
    print(f"\n[4/4] Regression results CSV → {CSV_PATH}")

    # -- Interpretation summary -----------------------------------------------

    print("\n  --- Interpretation summary ---")
    print()
    print("  Covariates that consistently predict influencer status:")

    for congress in CONGRESSES:
        sub = results_df[
            (results_df["congress"] == congress) &
            (results_df["outcome"] == "net_influence")
        ]
        if sub.empty:
            continue
        row = sub.iloc[0]
        sig_vars = []
        for col in ["coef_log_spend", "coef_log_bills", "coef_katz_centrality",
                    "coef_participation_coeff"]:
            pval_col = col.replace("coef_", "pval_")
            if pval_col in row and not pd.isna(row[pval_col]) and row[pval_col] < 0.10:
                sig_vars.append(
                    f"{col.replace('coef_','')} (β={row[col]:.3f}, p={row[pval_col]:.3f})"
                )
        label = "significant" if sig_vars else "none significant at p<0.10"
        print(f"  [{congress}th] net_influence: {label}")
        for v in sig_vars:
            print(f"    · {v}")

    print()
    print("  Validation complete.")
    print("=" * 75)

    log_f.close()
    sys.stdout = sys.__stdout__
    print(f"\nLog → {TXT_PATH}")


if __name__ == "__main__":
    main()
