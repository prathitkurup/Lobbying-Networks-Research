import pandas as pd
import re


df_clients = pd.read_csv(
    "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Prathit/Lobbying-Networks-Research/data/matched_company_subsidiaries.csv"
)

df_collected_clients = pd.read_csv(
    "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Prathit/Lobbying-Networks-Research/data/unique_clients.csv"
)

df_collected_clients["company_clean"] = df_collected_clients["company"]

#drop company column from df_collected_clients
df_collected_clients = df_collected_clients.drop(columns=["company"])

match_parent = df_clients.merge(
    df_collected_clients,
    left_on="company_clean",
    right_on="company_clean",
    how="inner",
    suffixes=("", "_client")
)
print("columns in match_parent:", match_parent.columns.tolist())
match_sub = df_clients.merge(
    df_collected_clients,
    left_on="subsidiary_name",
    right_on="company_clean",
    how="inner",
    suffixes=("", "_client")
)


merged = pd.concat([match_parent, match_sub], ignore_index=True)

merged = merged.drop_duplicates()

# # drop columns that aren't company_clean,subsidiary_name
# merged = merged[["company_clean", "subsidiary_name"]]


merged.to_csv(
    "/Users/vfigueroa/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/Prathit/Lobbying-Networks-Research/data/merged_clients_fortune.csv",
    index=False
)

print("Total matched rows:", len(merged))