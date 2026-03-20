import pandas as pd
from pathlib import Path


def load_bills_data(path):
    """Load opensecrets_lda_reports.csv and validate expected columns.

    Drops rows with a null bill_number (reports with no linked bills) so every
    row in the returned DataFrame represents a confirmed (firm, bill) pair.
    """
    df = pd.read_csv(path)
    check_columns(df, ["fortune_name", "bill_number", "amount_allocated"], path)
    return df.dropna(subset=["bill_number"]).copy()


def load_issues_data(path):
    """Load opensecrets_lda_issues.csv and validate expected columns."""
    df = pd.read_csv(path)
    check_columns(df, ["fortune_name", "issue_code", "amount_allocated"], path)
    return df


def load_lobby_firm_data(path):
    """Load opensecrets_lda_reports.csv for lobby-firm affiliation analysis.

    Keeps only rows with an external registrant (is_self_filer == False).
    is_self_filer is a boolean in the OpenSecrets CSV (True = self-filer).
    """
    df = pd.read_csv(path)
    check_columns(df, ["fortune_name", "registrant"], path)
    df = df[~df["is_self_filer"].fillna(False)].dropna(subset=["registrant", "fortune_name"])
    return df


def check_columns(df, required, path):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing}")
