"""
Build bill-company incidence (0/1) matrices for four Fortune tiers (500/100/50/30).

For each tier, two variants: with and without singleton bills (bills lobbied by only
one company in that tier). Filtered to 116th Congress, year 2019 only.

Outputs: data/affiliation/{f500,f100,f50,f30}/
  matrix_with_singletons.csv      raw 0/1, no header, no index
  matrix_no_singletons.csv        raw 0/1, no header, no index
  bill_index_with_singletons.csv
  bill_index_no_singletons.csv
  company_index_with_singletons.csv
  company_index_no_singletons.csv
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from config import DATA_DIR
from utils.data_loading import load_bills_data

FORTUNE_FILE = DATA_DIR / "archive" / "cleaning" / "fortune_canonical.csv"
OUT_DIR      = DATA_DIR / "affiliation"
YEAR         = 2019
TIERS        = [500, 100, 50, 30]


def load_fortune_set(n):
    """Return set of top-n Fortune companies by rank order in fortune_canonical.csv."""
    canon = pd.read_csv(FORTUNE_FILE)
    return set(canon["company"].iloc[:n])


def build_matrix(df):
    """Pivot to (bill_number x fortune_name) 0/1 incidence matrix."""
    return (df.assign(present=1)
              .pivot_table(index="bill_number", columns="fortune_name",
                           values="present", fill_value=0))


def write_variant(pivot, out_dir, suffix):
    """Write matrix (no header/index) + bill_index + company_index for one variant."""
    bills     = list(pivot.index)
    companies = list(pivot.columns)

    pivot.astype(int).to_csv(out_dir / f"matrix_{suffix}.csv", header=False, index=False)

    pd.DataFrame({"bill_number": bills,
                  "bill_idx": range(len(bills))}).to_csv(
        out_dir / f"bill_index_{suffix}.csv", index=False)

    pd.DataFrame({"company": companies,
                  "company_idx": range(len(companies))}).to_csv(
        out_dir / f"company_index_{suffix}.csv", index=False)

    print(f"    [{suffix}] {len(bills)} bills x {len(companies)} companies")


def main():
    df = load_bills_data(DATA_DIR / "opensecrets_lda_reports.csv")
    df = df[df["year"] == YEAR].drop_duplicates(subset=["fortune_name", "bill_number"])
    print(f"Loaded {len(df):,} (firm, bill) pairs for {YEAR}")

    for n in TIERS:
        tier_dir = OUT_DIR / f"f{n}"
        tier_dir.mkdir(parents=True, exist_ok=True)

        tier_companies = load_fortune_set(n)
        df_tier        = df[df["fortune_name"].isin(tier_companies)].copy()
        pivot          = build_matrix(df_tier)

        print(f"\n  f{n}:")
        write_variant(pivot, tier_dir, "with_singletons")

        pivot_ns = pivot[pivot.sum(axis=1) >= 2]
        write_variant(pivot_ns, tier_dir, "no_singletons")

    print("\nDone.")


if __name__ == "__main__":
    main()
