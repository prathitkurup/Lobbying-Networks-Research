"""
Build bill-company incidence (0/1) matrix for the 116th Congress.

No prevalence filter is applied — all bills are included regardless of
how many firms lobbied them.

Outputs (all to DATA_DIR):
  bill_company_matrix.csv  — rows = integer bill indices, cols = integer
                             company indices, values = 0/1
  bill_index.csv           — row_idx -> bill_number mapping
  company_index.csv        — col_idx -> fortune_name mapping
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from config import DATA_DIR
from utils.data_loading import load_bills_data

OUTPUT_MATRIX       = DATA_DIR / "bill_company_matrix.csv"
OUTPUT_MATRIX_PLAIN = DATA_DIR / "bill_company_matrix_plain.csv"
OUTPUT_BILL_IDX     = DATA_DIR / "bill_index.csv"
OUTPUT_COMP_IDX     = DATA_DIR / "company_index.csv"


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build binary bill-company incidence matrix."""
    # One row per (bill, company) pair — ignore duplicate filings
    pairs = df[["bill_number", "fortune_name"]].drop_duplicates()

    # Pivot to bill x company named matrix; fill missing with 0
    named = (
        pairs.assign(value=1)
        .pivot_table(
            index="bill_number",
            columns="fortune_name",
            values="value",
            fill_value=0,
        )
        .astype(int)
    )

    # pivot_table sorts index and columns alphabetically — gives reproducible order
    bills     = named.index.tolist()
    companies = named.columns.tolist()

    # Replace named index/columns with integer indices
    matrix = named.reset_index(drop=True)
    matrix.columns = pd.RangeIndex(len(companies))

    bill_index    = pd.DataFrame({"row_idx": range(len(bills)),     "bill_number":  bills})
    company_index = pd.DataFrame({"col_idx": range(len(companies)), "fortune_name": companies})

    return matrix, bill_index, company_index


def main():
    df = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")

    print(
        f"Loaded {len(df):,} rows  |  "
        f"Bills: {df['bill_number'].nunique():,}  |  "
        f"Companies: {df['fortune_name'].nunique():,}"
    )

    matrix, bill_index, company_index = build_matrix(df)

    n_bills, n_companies = matrix.shape
    n_ones = int(matrix.values.sum())
    density = n_ones / (n_bills * n_companies)

    print(f"\nMatrix shape : {n_bills} bills x {n_companies} companies")
    print(f"Non-zero (1s): {n_ones:,}  ({density:.2%} density)")

    # Save outputs
    matrix.to_csv(OUTPUT_MATRIX, index=True, index_label="row_idx")
    matrix.to_csv(OUTPUT_MATRIX_PLAIN, index=False, header=False)  # no row/col indices
    bill_index.to_csv(OUTPUT_BILL_IDX, index=False)
    company_index.to_csv(OUTPUT_COMP_IDX, index=False)

    print(f"\nOutputs saved:")
    print(f"  Matrix (indexed) -> {OUTPUT_MATRIX}")
    print(f"  Matrix (plain)   -> {OUTPUT_MATRIX_PLAIN}")
    print(f"  Bill index       -> {OUTPUT_BILL_IDX}")
    print(f"  Company index    -> {OUTPUT_COMP_IDX}")


if __name__ == "__main__":
    main()
