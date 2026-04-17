import pandas as pd
from d3graph import d3graph, vec2adjmat

df = pd.read_csv("fortune100_client_lob_bill_amount.csv")

top_bills = [
        # TOP 10 BILLS FROM OUR DATA:
        # "hr1668-116",
        # "s734-116",
        # "s748-116",
        # "hr1725-116",
        # "s604-116",
        # "s893-116",
        # "s918-116",
        # "hr1044-116",
        # "hr1644-116",
        # "hr5-116",

        # TECH BILLS:
        "hr1044-116",
        "hr1668-116",
        "hr5-116",
        "s734-116",
        "s748-116",
        "s788-116",
        "hr2013-116",
        "s189-116",
        "hr2820-116",
        "hr3494-116",

        # HEALTHCARE BILLS:
        # "s3-116",
        # "s1895-116",
        # "hr748-116",
        # "hr6201-116",
        # "hr3630-116",
        # "hr987-116",
        # "s1129-116",
        # "hr133-116",
        # "s4185-116",
        # "hr1425-116",

        # DEFENSE BILLS
        # "hr2500-116",
        # "s1790-116",
        # "hr6395-116",
        # "hr1158-116",
        # "s2474-116",
        # "hr5430-116",
        # "hr3014-116",
        # "hr6157-116",
        # "s881-116",
        # "hr1865-116"
    ]

# Filter to only top bills
df = df[df["bill_id"].str.lower().isin(top_bills)]

# d3graph columns: source, target, weight
df = df.rename(columns={
    "client_name": "source",
    "bill_id": "target",
    "amount": "weight"
})

# Ensure numeric, positive weights
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df = df[df["weight"] > 0]

print("Rows after filter:", df.shape[0])

# Build adjacency
adjmat = vec2adjmat(
    df["source"].tolist(),
    df["target"].tolist(),
    weight=df["weight"].tolist()
)
print("Adjacency shape:", adjmat.shape)
print("Non-zero edges:", (adjmat.values > 0).sum())

# Initialize and draw
d3 = d3graph()
d3.graph(adjmat)
d3.set_edge_properties(directed=False)
d3.set_node_properties(size='degree')

# Color mapping
source_nodes = set(df["source"])
target_nodes = set(df["target"])
color_map = {}
for s in source_nodes:
    color_map[s] = "#337aff"
for t in target_nodes:
    if t not in color_map:
        color_map[t] = "#ff5733"

nodes = adjmat.index.tolist()
node_colors = [color_map.get(node, "#999999") for node in nodes]
d3.set_node_properties(color=node_colors)
#print data stats
total_dollars = df["weight"].sum()
unique_companies = df["source"].nunique()
unique_bills = df["target"].nunique()
print(f"Total lobbying dollars spent: {total_dollars}")
print(f"Unique Fortune 100 companies: {unique_companies}")
print(f"Unique bills lobbied: {unique_bills}")
d3.show(
    title="Fortune 20 Companies Lobbying for Top 10 Tech Bills in 116th Congress",
    # Change the filepath to your desired location
    filepath="/Users/prathitkurup/Desktop/Game Theory/fortune20_lobbying_TECH_116th.html",
    figsize=(1200, 800)
)