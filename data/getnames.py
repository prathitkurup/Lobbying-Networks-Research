import pandas as pd

subs = pd.read_csv(
    "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Prathit/Lobbying-Networks-Research/data/fortune500_subsidiaries_ex21.csv"
)

companies = pd.read_csv(
    "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Prathit/Lobbying-Networks-Research/data/fortune_canonical.csv"
)

# -------------------------------------------------
# NORMALIZE BOTH SIDES (VERY IMPORTANT)
# -------------------------------------------------
companies["company_clean"] = companies["company"].str.upper().str.strip()
subs["company_clean"] = subs["matched_sec_company_name"].str.upper().str.strip()

# OPTIONAL but STRONGLY recommended: remove INC/CO/etc
import re

def normalize_name(s):
    if pd.isna(s):
        return s
    s = s.upper()
    s = re.sub(r'[.,]', '', s)
    s = re.sub(r'\b(INC|CORP|CORPORATION|CO|COMPANY|HOLDINGS|GROUP|LLC|LP)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

companies["company_clean"] = companies["company_clean"].apply(normalize_name)
subs["company_clean"] = subs["company_clean"].apply(normalize_name)

# -------------------------------------------------
# INNER JOIN (ONLY MATCHES)
# -------------------------------------------------
merged = companies.merge(
    subs,
    on="company_clean",
    how="inner"
)

# -------------------------------------------------
# KEEP ONLY WHAT YOU WANT
# -------------------------------------------------
print("Columns in merged file:", merged.columns.tolist())

result = (
    merged[["company_clean", "subsidiary_name"]]
    .dropna(subset=["subsidiary_name"])
    .drop_duplicates()
)

# drop columsn that aren't comany_clean or subsidiary_name
result = result[["company_clean", "subsidiary_name"]]
# -------------------------------------------------
# SAVE
# -------------------------------------------------
result.to_csv(
    "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Prathit/Lobbying-Networks-Research/data/matched_company_subsidiaries.csv",
    index=False
)

print("Rows in final file:", len(result))