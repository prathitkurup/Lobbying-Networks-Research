import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from d3graph import d3graph, vec2adjmat

df = pd.read_csv("../data/fortune500_lda_reports.csv")
print("number of bills", df["bill_id"].nunique())

# Find companies that share bills
# Group by bill to find all companies lobbying for each bill
bills_to_companies = df.groupby("bill_id")["client_name"].apply(list).reset_index()

# Create company-to-company edges based on shared bills
edges = []
for _, row in bills_to_companies.iterrows():
    companies = row["client_name"]
    # For each pair of companies on the same bill
    for i in range(len(companies)):
        for j in range(i + 1, len(companies)):
            if companies[i] != companies[j]:
                edges.append({
                    "source": companies[i],
                    "target": companies[j],
                    "bill_id": row["bill_id"]
            })

# Convert to dataframe and aggregate by company pairs
edges_df = pd.DataFrame(edges)
df = edges_df.groupby(["source", "target"]).size().reset_index(name="weight")

# Ensure numeric, positive weights
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df = df[df["weight"] > 0]

print("Rows after filter:", df.shape[0])

# save df to csv
df.to_csv("fortune500_shared_bills.csv", index=False)

# Build adjacency
adjmat = vec2adjmat(
    df["source"].tolist(),
    df["target"].tolist(),
    weight=df["weight"].tolist()
)
adjmat = adjmat + adjmat.T  # make undirected
print("Adjacency shape:", adjmat.shape)
#save adjacency matrix to csv, undirected graph
adjmat.to_csv("fortune500_shared_bills_adjmat.csv")
print("Non-zero edges:", (adjmat.values > 0).sum())

# Create Adj matrix text file for influence game PSNE computation
# Create a text file and write the adjacency matrix with tab separation between each element
with open("fortune500_lobbying_threshold_input.txt", "w") as f:
    for i in range(adjmat.shape[0]):
        for j in range(adjmat.shape[1]):
            f.write(str(adjmat.iloc[i, j]) + "\t")
        f.write("\n")

# TODO: Fix Threshold calculation
# For each company (row), sum the weights of edges, find average, and then divide by 1116 for each company
# for i in range(adjmat.shape[0]):
#     total_weight = adjmat.iloc[i, :].sum()
#     avg_weight = total_weight / (adjmat.shape[0] - 1)  # exclude self
#     # normalized_weight = avg_weight / 1116
#     company = adjmat.index[i]
#     print(f"{company}: {avg_weight:.4f}")

# # Initialize and draw D3 graph
# d3 = d3graph()
# d3.graph(adjmat)
# d3.set_edge_properties(directed=False)
# d3.set_node_properties(size='degree')

# # Color mapping
# source_nodes = set(df["source"])
# target_nodes = set(df["target"])
# color_map = {}
# for s in source_nodes:
#     color_map[s] = "#337aff"
# for t in target_nodes:
#     if t not in color_map:
#         color_map[t] = "#ff5733"

# nodes = adjmat.index.tolist()
# node_colors = [color_map.get(node, "#999999") for node in nodes]
# d3.set_node_properties(color=node_colors)

#print data stats
# total_edgeweights = df["weight"].sum()

# print(f"Total edge weights: {total_edgeweights}")
# d3.show(
#     title="Fortune 500 Companies Lobbying for Bills in 116th Congress",
#     # Change the filepath to your desired location
#     filepath="/Users/prathitkurup/Desktop/Game Theory/fortune500_lobbying_affiliation_d3.html",
#     figsize=(1200, 800)
# )

# Create NetworkX visualization of affiliation netowrk
print("\nCreating NetworkX visualization...")
G = nx.Graph()
# Add edges with weights
for _, row in df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# Select top 20 nodes by weighted degree (node strength)
strengths = {node: G.degree(node, weight="weight") for node in G.nodes()}
top_nodes = [n for n, _ in sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:20]]
# Induced subgraph on the top nodes (20 nodes with highest weighted degree) just for visualization
H = G.subgraph(top_nodes).copy()

# Layout
pos = nx.circular_layout(H)
# Node strengths and sizes
node_strength = np.array([H.degree(node, weight="weight") for node in H.nodes()])
node_sizes = 800 + 3000 * (node_strength - node_strength.min()) / (node_strength.max() - node_strength.min() + 1)
# Edge widths
edges = list(H.edges())
weights = np.array([H[u][v]["weight"] for u, v in edges]) if edges else np.array([])
edge_widths = 1 + 20 * (weights - (weights.min() if weights.size else 0)) / ((weights.max() - (weights.min() if weights.size else 0)) + 1) if weights.size else []
# Plotting
fig, ax = plt.subplots(figsize=(16, 16))
# Nodes
nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color="#337aff", alpha=1, linewidths=1.5, edgecolors="black", ax=ax)
# Weighted edges (only if present)
if edges:
    nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.7, edge_color="gray", ax=ax)
# Labels
nx.draw_networkx_labels(H, pos, font_size=12, font_weight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7), ax=ax)
nx.write_gml(G, "../visualizations/fortune_500_lobbying_affiliation_network.gml")

ax.set_title("Top 20 Fortune 500 Lobbying Affiliation Network (by weighted degree)", fontsize=18, fontweight="bold")
ax.axis("off")

plt.tight_layout()
plt.savefig( "../visualizations/fortune_500_lobbying_affiliation_network.png", dpi=300, bbox_inches="tight")
print("NetworkX visualization saved.")
plt.show()

# Get centraility measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
closeness_centrality = nx.closeness_centrality(G, distance="weight")
eigenvector_centrality = nx.eigenvector_centrality(G, weight="weight")

# print("\nCentrality for top 20 corporations (visualized):")
# for node in top_nodes:
#     print(f"{node}: Degree Centrality={degree_centrality[node]:.4f}, Betweenness Centrality={betweenness_centrality[node]:.4f}, Closeness Centrality={closeness_centrality[node]:.4f}, Eigenvector Centrality={eigenvector_centrality[node]:.4f}")

print("\nTop 5 with highest DEGREE centrality:")
for node, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {centrality:.4f}")

print("\nTop 5 with highest BETWEENNESS centrality:")
for node, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {centrality:.4f}")

print("\nTop 5 with highest CLOSENESS centrality:")
for node, centrality in sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {centrality:.4f}")

print("\nTop 5 with highest EIGENVECTOR centrality:")
for node, centrality in sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {centrality:.4f}")
