"""
Bill prevalence filtering: remove mega-bills lobbied by more than MAX_BILL_DF
firms before computing similarity metrics. See design_decisions.md §4.
"""

import pandas as pd


def filter_bills_by_prevalence(df, max_df, unit_col="bill_number"):
    """
    Remove rows where the number of unique firms lobbying the bill exceeds max_df.

    Parameters
    ----------
    df      : DataFrame with columns [fortune_name, <unit_col>]
    max_df  : int — maximum number of unique firms per bill (inclusive).
              Bills lobbied by more than max_df firms are excluded.
    unit_col: column name for the lobbying unit (bill_number or issue_code)

    Returns
    -------
    Filtered copy of df.  The total-budget denominator is NOT recomputed here;
    callers that need budget-consistent fracs should compute totals BEFORE
    calling this function (cosine/RBO) or AFTER (affiliation).
    """
    firms_per_unit = df.groupby(unit_col)["fortune_name"].nunique()
    keep = firms_per_unit[firms_per_unit <= max_df].index
    n_removed = (firms_per_unit > max_df).sum()
    pct = 100 * n_removed / len(firms_per_unit)
    print(f"\n[prevalence filter] {unit_col}: removed {n_removed} / "
          f"{len(firms_per_unit)} units with df > {max_df} ({pct:.1f}%)")
    return df[df[unit_col].isin(keep)].copy()


def prevalence_summary(df, unit_col="bill_number", thresholds=(10, 20, 30, 50, 100)):
    """
    Print a breakdown of the firms-per-unit distribution and show how many
    units would be removed at each threshold. Useful for threshold selection.
    """
    firms_per_unit = df.groupby(unit_col)["fortune_name"].nunique().sort_values(ascending=False)
    n_total = len(firms_per_unit)
    n_firms = df["fortune_name"].nunique()

    print(f"\n-- {unit_col} prevalence distribution --")
    print(f"  Total unique {unit_col}s: {n_total:,}")
    print(f"  Total unique firms:    {n_firms}")
    print(f"  {'Threshold':>10}  {'Removed':>8}  {'% Removed':>10}  "
          f"{'% of firms (max)':>17}")
    for t in thresholds:
        removed = (firms_per_unit > t).sum()
        pct_removed = 100 * removed / n_total
        max_prev = firms_per_unit[firms_per_unit > t].max() if removed > 0 else 0
        print(f"  {t:>10}  {removed:>8}  {pct_removed:>9.1f}%  {max_prev:>17}")

    print(f"\n  Top 20 most-prevalent {unit_col}s:")
    for unit, cnt in firms_per_unit.head(20).items():
        bar = "█" * min(int(cnt / n_firms * 40), 40)
        print(f"  {unit:<30}  {cnt:>4} firms  {bar}")
