"""
[ARCHIVED] Build bill-company incidence matrix (presence/absence).

Two variants:
  - Fortune 500 (original): bill_company_matrix.csv, bill_company_matrix_plain.csv,
    bill_index.csv, company_index.csv
  - Fortune 100, no singletons: bill_company_matrix_f100.csv,
    bill_company_matrix_plain_f100.csv, bill_index_f100.csv, company_index_f100.csv

Outputs (archived): data/archive/network_edges/
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from config import DATA_DIR
from utils.data_loading import load_bills_data

ARCHIVE      = DATA_DIR / "archive"
FORTUNE_N    = 100   # top-N companies by Fortune rank to include
FORTUNE_FILE = DATA_DIR / "archive" / "cleaning" / "fortune_canonical.csv"


def load_fortune_n(n):
    """Return set of the top-n Fortune companies by rank order in fortune_canonical.csv."""
    canon = pd.read_csv(FORTUNE_FILE)
    return set(canon["company"].iloc[:n])


def build_matrix(df):
    """Pivot (bill_number x fortune_name) 0/1 incidence matrix."""
    return (df.assign(present=1)
              .pivot_table(index="bill_number", columns="fortune_name",
                           values="present", fill_value=0))


def drop_singleton_bills(pivot):
    """Drop bills lobbied by fewer than 2 companies (no co-lobbying signal)."""
    return pivot[pivot.sum(axis=1) >= 2]


def main():
    df = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    df = df.drop_duplicates(subset=["fortune_name", "bill_number"])

    # --- Fortune 100 variant (no prevalence filter, no singletons) ---
    f100 = load_fortune_n(FORTUNE_N)
    df_f100 = df[df["fortune_name"].isin(f100)].copy()

    pivot_f100 = build_matrix(df_f100)
    pivot_f100 = drop_singleton_bills(pivot_f100)

    bills_f100    = list(pivot_f100.index)
    companies_f100 = list(pivot_f100.columns)
    bill_idx_f100  = pd.DataFrame({"bill_number": bills_f100,
                                   "bill_idx": range(len(bills_f100))})
    comp_idx_f100  = pd.DataFrame({"company": companies_f100,
                                   "company_idx": range(len(companies_f100))})

    net_dir = ARCHIVE / "network_edges"
    pivot_f100.to_csv(net_dir / "bill_company_matrix_f100.csv")
    pivot_f100.reset_index().to_csv(net_dir / "bill_company_matrix_plain_f100.csv",
                                    index=False)
    bill_idx_f100.to_csv(net_dir / "bill_index_f100.csv", index=False)
    comp_idx_f100.to_csv(net_dir / "company_index_f100.csv", index=False)
    print(f"F100 matrix: {len(bills_f100)} bills x {len(companies_f100)} companies"
          f" -> {net_dir}")


if __name__ == "__main__":
    main()
