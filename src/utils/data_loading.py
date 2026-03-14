import pandas as pd
from pathlib import Path


def load_bills_data(path):
    """Load fortune500_lda_reports.csv and validate expected columns."""
    df = pd.read_csv(path)
    check_columns(df, ["client_name", "bill_id", "amount"], path)
    return df


def load_issues_data(path):
    """Load fortune500_lda_issues.csv and validate expected columns."""
    df = pd.read_csv(path)
    check_columns(df, ["client_name", "general_issue_code", "amount"], path)
    return df


def load_lobby_firm_data(path):
    """
    Load fortune500_lda_reports.csv for lobby-firm affiliation analysis.
    """
    df = pd.read_csv(path)
    check_columns(df, ["client_name", "registrant_id"], path)
    # Keep only rows with an external registrant (is_self_filer != 't').
    mask = (
        (df["is_self_filer"] == "f") |
        df["is_self_filer"].isna() |
        (df["is_self_filer"] == "")
    )
    df = df[mask].dropna(subset=["registrant_id", "client_name"])
    return df


def check_columns(df, required, path):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing}")
