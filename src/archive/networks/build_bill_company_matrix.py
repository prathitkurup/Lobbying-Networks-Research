"""
[ARCHIVED] Build bill-company incidence matrix (presence/absence).

Outputs (archived): data/archive/network_edges/bill_company_matrix.csv,
bill_company_matrix_plain.csv, bill_index.csv, company_index.csv
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from config import DATA_DIR, MAX_BILL_DF
from utils.data_loading import load_bills_data
from utils.filtering import filter_bills_by_prevalence

ARCHIVE = DATA_DIR / "archive"


def main():
    df = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    df = df.drop_duplicates(subset=["fortune_name", "bill_number"])
    if MAX_BILL_DF is not None:
        df = filter_bills_by_prevalence(df, MAX_BILL_DF, unit_col="bill_number")

    pivot = (df.assign(present=1)
               .pivot_table(index="bill_number", columns="fortune_name",
                            values="present", fill_value=0))

    bills    = list(pivot.index)
    companies = list(pivot.columns)
    bill_idx  = pd.DataFrame({"bill_number": bills, "bill_idx": range(len(bills))})
    comp_idx  = pd.DataFrame({"company": companies, "company_idx": range(len(companies))})

    net_dir = ARCHIVE / "network_edges"
    pivot.to_csv(net_dir / "bill_company_matrix.csv")
    pivot.reset_index().to_csv(net_dir / "bill_company_matrix_plain.csv", index=False)
    bill_idx.to_csv(net_dir / "bill_index.csv", index=False)
    comp_idx.to_csv(net_dir / "company_index.csv", index=False)
    print(f"Matrix: {len(bills)} bills x {len(companies)} companies -> {net_dir}")


if __name__ == "__main__":
    main()
