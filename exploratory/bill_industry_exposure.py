"""
Bill industry exposure statistics using CRP catcodes.

For each Fortune 500 company in the 116th Congress, derives a primary industry
from the catcode in lob_lobbying.txt (CRP-assigned, exogenous to issue content).
Each bill is mapped to the catcode sector letter of its policy domain.

Outputs:
  data/fortune_primary_industry_116.csv  — fortune_name, catcode, sector_letter,
                                           sector_name, total_spend
  Prints per-bill table: N_universe, N_actual, coverage.
"""

import csv
import pandas as pd

LOBBYING = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
            "Lobbying-Networks-Research/data/OpenSecrets/lob_lobbying.txt")
DATA_116 = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
            "Lobbying-Networks-Research/data/congress/116")
DATA_115 = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
            "Lobbying-Networks-Research/data/congress/115")
DATA_ROOT = ("/sessions/serene-lucid-franklin/mnt/Independent Study/"
             "Lobbying-Networks-Research/data")

# CRP sector-letter → descriptive name (verified against Fortune company samples)
SECTOR_NAMES = {
    "A": "Agribusiness",
    "B": "Construction & Engineering",
    "C": "Communications, Electronics & Tech",
    "D": "Defense & Aerospace",
    "E": "Energy & Natural Resources",
    "F": "Finance, Insurance & Real Estate",
    "G": "General Business & Retail",
    "H": "Health",
    "M": "Manufacturing & Distribution",
    "T": "Transportation & Automotive",
    "Y": "Other / Miscellaneous",
}

# Bill → catcode sector letter for the 12 bills of interest
BILL_SECTOR = {
    "H.R.1044": "C",   # Tech: Fairness for High-Skilled Immigrants Act
    "H.R.1644": "C",   # Tech: Save the Internet Act (Net Neutrality)
    "H.R.1628": "H",   # Healthcare: American Health Care Act
    "S.1895":   "H",   # Healthcare: Lower Health Care Costs Act
    "S.1460":   "E",   # Energy: Energy & Natural Resources Act
    "H.R.360":  "E",   # Energy: Cyber Sense Act
    "H.R.2810": "D",   # Defense: NDAA FY2018
    "H.R.2500": "D",   # Defense: NDAA FY2020
    "H.R.10":   "F",   # Finance: Financial CHOICE Act
    "H.R.1994": "F",   # Finance: SECURE Act
    "S.1405":   "T",   # Transportation: FAA Reauthorization
    "S.2302":   "T",   # Transportation: America's Transportation Infrastructure Act
}

BILLS = [
    (116, "H.R.1044", "Technology",      "Fairness for High-Skilled Immigrants Act"),
    (116, "H.R.1644", "Technology",      "Save the Internet Act"),
    (115, "H.R.1628", "Healthcare",      "American Health Care Act"),
    (116, "S.1895",   "Healthcare",      "Lower Health Care Costs Act"),
    (115, "S.1460",   "Energy",          "Energy & Natural Resources Act of 2017"),
    (116, "H.R.360",  "Energy",          "Cyber Sense Act"),
    (115, "H.R.2810", "Defense",         "NDAA FY2018"),
    (116, "H.R.2500", "Defense",         "NDAA FY2020"),
    (115, "H.R.10",   "Finance/Banking", "Financial CHOICE Act"),
    (116, "H.R.1994", "Finance/Banking", "SECURE Act"),
    (115, "S.1405",   "Transportation",  "FAA Reauthorization Act of 2017"),
    (116, "S.2302",   "Transportation",  "America's Transportation Infrastructure Act"),
]

# ── Load processed reports (116th for catcode join; 115th for actual lobbyist counts) ──
rep_116 = pd.read_csv(f"{DATA_116}/opensecrets_lda_reports.csv")
rep_115 = pd.read_csv(f"{DATA_115}/opensecrets_lda_reports.csv")

uid_to_fortune_116 = rep_116.set_index("uniq_id")["fortune_name"].to_dict()
our_uids_upper     = {u.upper(): u for u in uid_to_fortune_116}

# ── Extract catcode + amount per (fortune_name, catcode) from lob_lobbying ──
records = []
with open(LOBBYING, encoding="latin-1") as f:
    reader = csv.reader(f, quotechar="|", delimiter=",", doublequote=False)
    for row in reader:
        if len(row) < 9:
            continue
        uid_upper = row[0].strip().upper()
        if uid_upper not in our_uids_upper:
            continue
        catcode = row[8].strip()
        if len(catcode) != 5 or not catcode[0].isalpha():
            continue
        try:
            amount = float(row[7].strip()) if row[7].strip() else 0.0
        except ValueError:
            amount = 0.0
        fortune_name = uid_to_fortune_116[our_uids_upper[uid_upper]]
        records.append({"fortune_name": fortune_name, "catcode": catcode, "amount": amount})

cat_df = pd.DataFrame(records)

# Primary catcode per company = catcode with highest total spend
spend_by_cat = (
    cat_df.groupby(["fortune_name", "catcode"])["amount"]
    .sum()
    .reset_index()
)
idx = spend_by_cat.groupby("fortune_name")["amount"].idxmax()
primary = spend_by_cat.loc[idx].copy()
primary.columns = ["fortune_name", "primary_catcode", "total_spend"]
primary["sector_letter"] = primary["primary_catcode"].str[0]
primary["sector_name"]   = primary["sector_letter"].map(SECTOR_NAMES).fillna("Other")
primary = primary.sort_values("fortune_name").reset_index(drop=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = f"{DATA_ROOT}/fortune_primary_industry_116.csv"
primary[["fortune_name","primary_catcode","sector_letter","sector_name","total_spend"]].to_csv(
    out_csv, index=False
)
print(f"Saved → {out_csv}")
print(f"  {len(primary)} Fortune companies  |  {primary['sector_letter'].nunique()} sectors\n")

print("Companies per sector (116th Congress):")
sec_counts = primary.groupby(["sector_letter","sector_name"]).size().reset_index(name="n_companies")
for _, row in sec_counts.iterrows():
    print(f"  {row['sector_letter']}  {row['sector_name']:<40} {row['n_companies']:>3} companies")

# ── Build sector → set of fortune_names lookup ────────────────────────────────
sector_universe = primary.groupby("sector_letter")["fortune_name"].apply(set).to_dict()

# ── Bill exposure stats ────────────────────────────────────────────────────────
reports = {115: rep_115, 116: rep_116}

print(f"\n{'Bill':<12} {'Cong':>5}  {'Sector':<16} {'CRP Sec':>8}  "
      f"{'Sector Name':<38}  {'N_univ':>6}  {'N_act':>6}  {'Coverage':>9}  Title")
print("─" * 135)

for (congress, bill, sector, title) in BILLS:
    sec_letter  = BILL_SECTOR.get(bill, "?")
    universe    = sector_universe.get(sec_letter, set())
    n_universe  = len(universe)
    n_actual    = reports[congress][reports[congress]["bill_number"] == bill]["fortune_name"].nunique()
    coverage    = n_actual / n_universe if n_universe > 0 else float("nan")
    sec_name    = SECTOR_NAMES.get(sec_letter, "?")

    print(f"{bill:<12} {congress:>5}  {sector:<16} {sec_letter:>8}  "
          f"{sec_name:<38}  {n_universe:>6}  {n_actual:>6}  {coverage:>8.1%}  {title}")
