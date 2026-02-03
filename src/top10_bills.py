# import pandas as pd

# # Load CSV
# df = pd.read_csv("fortune100_client_lob_bill_amount.csv")

# # Aggregate amounts by bill_id
# top_10 = (
#     df.groupby("bill_id")["amount"]
#       .sum()
#       .sort_values(ascending=False)
#       .head(10)
# )
# print("TOP 10 AGGREGATED")
# print(top_10)



import pandas as pd

# Load the data
A = pd.read_csv("TEST_fortune100_client_lob_bill_amount.csv")
B = pd.read_csv("lobbyview_issue_text.csv")

codes = {"CPI", "CSP", "TEC", "SCI"}

# Filter B to reports with any of the desired issue codes
B_xyz = B[B["general_issue_code"].isin(codes)]

# Join with A
A_xyz = A.merge(
    B_xyz[["report_uuid"]],
    on="report_uuid",
    how="inner"
)

# Aggregate
top_10_xyz_bills = (
    A_xyz.groupby("bill_id")["amount"]
         .sum()
         .sort_values(ascending=False)
         .head(10)
)

print(top_10_xyz_bills)