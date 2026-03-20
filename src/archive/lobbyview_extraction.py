"""
Extract bill-level lobbying activity for Fortune 500 companies from LobbyView.
Allocates report-level spending evenly across all associated bills and maps
client names to canonical Fortune 500 identities.

Run: python lobbyview_extraction.py
"""

import json
import re
import pandas as pd
from config import LOBBYVIEW_DIR, NAME_MAPPING, OUTPUT_CSV, OUTPUT_ISSUES_CSV, TARGET_CONGRESS, CONGRESS_FILING_YEARS


# -- Name mapping --

def load_name_mapping(path):
    with open(path) as f:
        return json.load(f)

def build_lookup(mapping):
    """Flat dict from any normalized variant/subsidiary to canonical name."""
    lookup = {}
    for canonical, entry in mapping.items():
        lookup[_norm(canonical)] = canonical
        for name in entry.get("variations", []) + entry.get("subsidiaries", []):
            lookup[_norm(name)] = canonical
    return lookup

def _norm(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.upper().strip())


# -- Data loading --

def load_lobbyview():
    clients = pd.read_csv(
        LOBBYVIEW_DIR / "lobbyview_clients.csv",
        usecols=["lob_id", "client_name"]
    )
    reports = pd.read_csv(
        LOBBYVIEW_DIR / "lobbyview_reports.csv",
        usecols=["report_uuid", "lob_id", "amount", "filing_year",
                 "is_self_filer", "registrant_id", "registrant_name"]
    )
    issues = pd.read_csv(
        LOBBYVIEW_DIR / "lobbyview_issue_text.csv",
        usecols=["report_uuid", "general_issue_code", "bill_id_agg"]
    )
    bills = pd.read_csv(
        LOBBYVIEW_DIR / "lobbyview_bills.csv",
        usecols=["bill_id", "congress_number", "bioguide_id"]
    )
    return clients, reports, issues, bills


# -- Pipeline steps --

def explode_bills(issues):
    """Clean and explode the aggregated bill column to one row per bill."""
    df = issues.copy()
    df["bill_id_agg"] = (
        df["bill_id_agg"].fillna("").astype(str)
        .str.replace(r'[{}\"]', "", regex=True)
        .str.replace(r"^\s*nan\s*$", "", regex=True)
    )
    df = df.assign(bill_id=df["bill_id_agg"].str.split(",")).explode("bill_id")
    df["bill_id"] = df["bill_id"].str.strip()
    return df[df["bill_id"] != ""]


def filter_congress(issues, bills, congress):
    """Join bill metadata and keep only records from the target Congress."""
    df = issues.merge(
        bills[["bill_id", "congress_number", "bioguide_id"]],
        on="bill_id", how="left"
    )
    return df[df["congress_number"] == congress].copy()


def join_reports(df, reports):
    """Attach report metadata and split spending evenly across associated bills."""
    cols = ["report_uuid", "lob_id", "amount", "filing_year",
            "is_self_filer", "registrant_id", "registrant_name"]
    df = df.merge(reports[cols], on="report_uuid", how="left")

    bill_counts = df.groupby("report_uuid")["bill_id"].nunique().rename("num_bills")
    df = df.merge(bill_counts, on="report_uuid", how="left")
    df["amount"] = (df["amount"] / df["num_bills"]).fillna(0)
    return df


def map_clients(df, clients, lookup):
    """Join client names and map to canonical Fortune 500 identities."""
    df = df.merge(clients[["lob_id", "client_name"]], on="lob_id", how="left")
    df["fortune_name"] = df["client_name"].map(lambda x: lookup.get(_norm(x)))
    return df[df["fortune_name"].notna()].copy()


def build_output(df):
    return (
        df[[
            "fortune_name", "lob_id", "report_uuid", "bill_id",
            "amount", "filing_year", "is_self_filer",
            "registrant_id", "registrant_name"
        ]]
        .rename(columns={"fortune_name": "client_name"})
        .drop_duplicates()
    )


# -- Issue-level pipeline --

def extract_by_issue(issues, reports, clients, lookup):
    """
    Produce a report-level dataset grouped by issue code rather than bill.
    Filters to the 116th Congress by filing year (2019-2020) since there is
    no bill join to provide a congress_number. Spending is split evenly across
    all issue codes within a report.
    """
    # One row per (report, issue_code) — drop rows with no issue code
    df = issues[["report_uuid", "general_issue_code"]].dropna().drop_duplicates()

    # Join report metadata and filter to 116th Congress filing years
    report_cols = ["report_uuid", "lob_id", "amount", "filing_year",
                   "is_self_filer", "registrant_id", "registrant_name"]
    df = df.merge(reports[report_cols], on="report_uuid", how="left")
    df = df[df["filing_year"].isin(CONGRESS_FILING_YEARS)].copy()

    # Split spending evenly across issue codes within each report
    issue_counts = df.groupby("report_uuid")["general_issue_code"].nunique().rename("num_issues")
    df = df.merge(issue_counts, on="report_uuid", how="left")
    df["amount"] = (df["amount"] / df["num_issues"]).fillna(0)

    # Map to Fortune 500
    df = df.merge(clients[["lob_id", "client_name"]], on="lob_id", how="left")
    df["fortune_name"] = df["client_name"].map(lambda x: lookup.get(_norm(x)))
    df = df[df["fortune_name"].notna()].copy()

    return (
        df[[
            "fortune_name", "lob_id", "report_uuid", "general_issue_code",
            "amount", "filing_year", "is_self_filer",
            "registrant_id", "registrant_name"
        ]]
        .rename(columns={"fortune_name": "client_name"})
        .drop_duplicates()
    )


# -- Validation --

def validate(bills_df, issues_df, mapping):
    """Print summaries for both outputs and flag companies missing from each."""
    unit_col = {"bill_id": bills_df, "general_issue_code": issues_df}

    for col, df in unit_col.items():
        label = "Bill-level" if col == "bill_id" else "Issue-level"
        print(f"\n-- {label} extraction (Congress {TARGET_CONGRESS}) --")
        print(f"  Records:   {len(df):,}")
        print(f"  Companies: {df['client_name'].nunique()}")
        print(f"  {col.replace('_', ' ').title()}s: {df[col].nunique()}")
        print(f"  Spend:     ${df['amount'].sum():,.0f}")

    missing_bills  = [c for c in mapping if c not in bills_df["client_name"].values]
    missing_issues = [c for c in mapping if c not in issues_df["client_name"].values]
    only_in_issues = [c for c in missing_bills if c not in missing_issues]

    print(f"\n-- Coverage --")
    print(f"  Bill-level only:            {bills_df['client_name'].nunique()} companies")
    print(f"  Issue-level only:           {issues_df['client_name'].nunique()} companies")
    print(f"  Gained from issue-level:    {len(only_in_issues)} companies")
    print(f"  Missing from both:          {len(missing_issues)} companies")
    if only_in_issues:
        print(f"  Recovered (issue not bill): {', '.join(only_in_issues[:10])}" +
              (" ..." if len(only_in_issues) > 10 else ""))


# -- Main --

def main():
    mapping = load_name_mapping(NAME_MAPPING)
    lookup  = build_lookup(mapping)

    clients, reports, issues, bills = load_lobbyview()

    # Bill-level extraction
    issues_exploded = explode_bills(issues)
    issues_filtered = filter_congress(issues_exploded, bills, TARGET_CONGRESS)
    issues_reports  = join_reports(issues_filtered, reports)
    fortune_only    = map_clients(issues_reports, clients, lookup)
    bills_final     = build_output(fortune_only)
    bills_final.to_csv(OUTPUT_CSV, index=False)

    # Issue-level extraction
    issues_final = extract_by_issue(issues, reports, clients, lookup)
    issues_final.to_csv(OUTPUT_ISSUES_CSV, index=False)

    validate(bills_final, issues_final, mapping)


if __name__ == "__main__":
    main()
