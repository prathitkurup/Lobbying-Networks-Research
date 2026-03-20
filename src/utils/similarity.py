"""
Shared similarity utilities for RBO and cosine-based network construction.

Functions
---------
rbo_score(l1, l2, p)
    Rank-Biased Overlap between two ranked bill lists.

build_ranked_lists(df, top_bills)
    Build per-firm bill rankings (by spend) from aggregated data.

build_frac_matrix(df)
    Build a (firms x bills) DataFrame of portfolio-share fracs.

aggregate_per_firm_bill(df)
    Collapse multiple rows per (fortune_name, bill_number) by summing
    amount_allocated.

compute_zero_budget_fracs(df)
    Add frac column; exclude zero-budget firms with a warning.

References
----------
Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure for
indefinite rankings. ACM Trans. Inf. Syst., 28(4), 1-38.
"""

import numpy as np
import pandas as pd


# -- Aggregation helpers -------------------------------------------------------

def aggregate_per_firm_bill(df):
    """
    Collapse multiple rows per (fortune_name, bill_number) by summing
    amount_allocated.
    opensecrets_extraction.py allocates report spend equally across bills, so a
    firm filing multiple reports on the same bill accumulates multiple rows.
    Summing gives the true total allocated spend per (firm, bill).
    """
    return df.groupby(
        ["fortune_name", "bill_number"], as_index=False
    )["amount_allocated"].sum()


def compute_zero_budget_fracs(df):
    """
    Add per-firm portfolio-share fracs (amount_allocated / total_budget).
    Firms with zero total budget are excluded with a warning.
    Returns the enriched DataFrame (only non-zero-budget firms).

    Validates that fracs sum to 1.0 per firm; raises ValueError on failure.
    """
    company_totals = df.groupby("fortune_name")["amount_allocated"].sum().rename("total_budget")
    df = df.merge(company_totals, on="fortune_name", how="left")

    zero_budget = company_totals[company_totals == 0].index.tolist()
    if zero_budget:
        print(f"  Warning: {len(zero_budget)} firm(s) excluded (zero total budget).")
        df = df[df["total_budget"] > 0].copy()

    df["frac"] = df["amount_allocated"] / df["total_budget"]

    frac_sums = df.groupby("fortune_name")["frac"].sum()
    bad = frac_sums[~frac_sums.between(0.999, 1.001)]
    if not bad.empty:
        raise ValueError(
            f"frac values do not sum to 1.0 for {len(bad)} firm(s): "
            f"{bad.to_dict()}\n"
            "Check opensecrets_extraction.py for changes that may have broken allocation logic."
        )
    return df


# -- RBO -----------------------------------------------------------------------

def rbo_score(l1, l2, p=0.90):
    """
    Rank-Biased Overlap (Webber et al. 2010), truncated min estimate.

    Compares two ranked lists up to the length of the shorter list.
    Agreement at higher ranks (index 0) is weighted more heavily.
    p controls decay rate: p=0.90 gives ~65% weight to top-10 items;
    p=0.98 gives a shallower decay (top-50 items dominate).

    Parameters
    ----------
    l1, l2 : list
        Ordered sequences of items, index 0 = highest priority.
    p      : float in (0, 1)
        Persistence parameter.

    Returns
    -------
    float in [0, 1]
    """
    s = min(len(l1), len(l2))
    if s == 0:
        return 0.0
    set1: set = set()
    set2: set = set()
    total = 0.0
    for d in range(1, s + 1):
        set1.add(l1[d - 1])
        set2.add(l2[d - 1])
        total += (p ** (d - 1)) * (len(set1 & set2) / d)
    return (1.0 - p) * total


def build_ranked_lists(df, top_bills=100):
    """
    Build a dict mapping each firm to its bill priority ranking.

    Bills are ranked by allocated spend (descending). Only bills with
    positive spend are included; i.e., the firm must have actively
    lobbied the bill.

    Parameters
    ----------
    df        : DataFrame with columns [fortune_name, bill_number, amount_allocated].
                Should already be aggregated to one row per (firm, bill).
    top_bills : int
                Maximum list length per firm (0 = no truncation).
                Truncating focuses RBO on high-spend priorities and
                prevents noise from the many low-spend bills that may
                only coincidentally appear in both firms' lists.

    Returns
    -------
    dict : {firm_name: [bill_number, bill_number, ...]}  (ordered, highest first)
    """
    ranked = {}
    for firm, grp in df.groupby("fortune_name"):
        bills_sorted = (grp[grp["amount_allocated"] > 0]
                        .sort_values("amount_allocated", ascending=False)["bill_number"]
                        .tolist())
        if top_bills and top_bills > 0:
            bills_sorted = bills_sorted[:top_bills]
        ranked[firm] = bills_sorted
    return ranked


# -- Cosine matrix -------------------------------------------------------------

def build_frac_matrix(df):
    """
    Build a (firms x bills) portfolio-share matrix for cosine similarity.

    Parameters
    ----------
    df : DataFrame with columns [fortune_name, bill_number, frac].
         Should already have fracs computed (via compute_zero_budget_fracs).

    Returns
    -------
    pivot : DataFrame  shape (n_firms, n_bills), zero-filled for missing pairs.
    firms : list of firm names (row order).
    bills : list of bill numbers (column order).

    Notes
    -----
    Since fracs are non-negative, cosine similarity will be in [0, 1].
    This is closely related to Pearson correlation of un-centered vectors.
    """
    pivot = df.pivot_table(
        index="fortune_name", columns="bill_number",
        values="frac", fill_value=0.0
    )
    return pivot, list(pivot.index), list(pivot.columns)
