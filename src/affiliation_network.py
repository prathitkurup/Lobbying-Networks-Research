'''
Prathit Kurup, Victoria Figueroa
02/17/2026

Build a company-to-company affiliation network based on shared bill lobbying.
Produce adjacency matrix (and threshold calculations for PSNE), 
NetworkX graph, and centrality measures.
'''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from d3graph import d3graph, vec2adjmat

DATA_PATH = "../data/fortune500_lda_reports.csv"
EDGE_OUTPUT_PATH = "fortune500_shared_bills.csv"
ADJMAT_OUTPUT_PATH = "fortune500_shared_bills_adjmat.csv"
PSNE_ADJMAT_TXT_PATH = "./influence_game_psne_calculation/fortune500_psne_game_input.txt"
GML_OUTPUT_PATH = "../visualizations/fortune_500_lobbying_affiliation_network.gml"
PNG_OUTPUT_PATH = "../visualizations/fortune_500_lobbying_affiliation_network.png"

# CREATE AFFILIATION NETWORK AND ADJACENCY MATRIX
def company_bill_edges(dataset_df, edge_list_path=EDGE_OUTPUT_PATH):
    """Create pairwise company edges from shared bills. Each shared bill generates a complete subgraph."""
    # For each bill, collect all companies that lobbied on it
    bill_company_df = dataset_df.groupby("bill_id")["client_name"].apply(list).reset_index(name="companies")
    
    # Generate edges for each bill's company list (complete subgraph)
    edge_records = []
    for _, row in bill_company_df.iterrows():
        companies = row["companies"]
        bill_id = row["bill_id"]
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                if companies[i] != companies[j]:
                    edge_records.append({
                        "source": companies[i],
                        "target": companies[j],
                        "bill_id": bill_id
                    })
    
    # Aggregate shared bills into weighted edges, where weight = number of shared bills
    edges_df = pd.DataFrame(edge_records)
    aggregate_edges = edges_df.groupby(["source", "target"]).size().reset_index(name="weight")
    aggregate_edges["weight"] = pd.to_numeric(aggregate_edges["weight"], errors="coerce")
    aggregate_edges = aggregate_edges[aggregate_edges["weight"] > 0]
    aggregate_edges.to_csv(edge_list_path, index=False)
    # print(aggregate_edges["weight"].describe())

    return aggregate_edges

def build_adjacency_matrix(edge_df, adjmat_output_path=ADJMAT_OUTPUT_PATH):
    """Construct undirected weighted adjacency matrix."""
    adjmat = vec2adjmat(
        edge_df["source"].tolist(),
        edge_df["target"].tolist(),
        weight=edge_df["weight"].tolist()
    )
    adjmat = adjmat + adjmat.T  # Enforce adjmat symmetry (make into undirected graph)
    adjmat.to_csv(adjmat_output_path)
    return adjmat

def build_d3_graph(adjmat: pd.DataFrame):
    """Interactive D3 visualization."""
    d3 = d3graph()
    d3.graph(adjmat)
    d3.set_edge_properties(directed=False)
    d3.set_node_properties(size='degree')


# COMPUTE THRESHOLDS FOR INFLUENCE GAME PSNE CALCULATION
def compute_thresholds(adjmat, percentile=75):
    """
    Influence threshold computation for PSNE calculation using percentiles, where
        threshold_i = percentile_p({ w_ij | w_ij > 0 })
    """
    thresholds = {}

    for i in range(adjmat.shape[0]):
        company = adjmat.index[i]
        row_vals = adjmat.iloc[i, :].values
        nonzero_weights = row_vals[row_vals > 0]

        # If no connections, set threshold to 0; otherwise compute percentile
        if len(nonzero_weights) == 0:
            threshold = 0.0
        else:
            threshold = np.percentile(nonzero_weights, percentile)
        thresholds[company] = threshold

    return thresholds

def save_psne_input_with_threshold(adjmat, thresholds, threshold_txt_path=PSNE_ADJMAT_TXT_PATH):
    """Save adjacency matrix as tab-separated text (for PSNE calculation input format) with thresholds in last column."""
    with open(threshold_txt_path, "w") as f:
        for i in range(adjmat.shape[0]):
            company = adjmat.index[i]
            row_vals = [str(adjmat.iloc[i, j]) for j in range(adjmat.shape[1])]
            row_vals.append(str(thresholds[company]))
            f.write("\t".join(row_vals) + "\n")


# CONSTRUCT NETWORKX GRAPH
def build_networkx_graph(edge_df, gml_output_path=GML_OUTPUT_PATH):
    """Build NetworkX graph from weighted edge list."""
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    nx.write_gml(G, gml_output_path)
    return G

def extract_top_k_subgraph(G, k=20):
    """Extract subgraph of top-k nodes by weighted degree (node strength)."""
    strengths = {node: G.degree(node, weight="weight") for node in G.nodes()}
    top_nodes = [
        node for node, _ in
        sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:k]
    ]
    H = G.subgraph(top_nodes).copy()
    return H


# VISUALIZATION
def plot_affiliation_network(H, png_path=PNG_OUTPUT_PATH, gml_path=GML_OUTPUT_PATH):
    """Visualize top-k affiliation network."""
    pos = nx.circular_layout(H)

    # Node sizing (weighted degree)
    strengths = np.array([H.degree(n, weight="weight") for n in H.nodes()])
    node_sizes = 800 + 3000 * (strengths - strengths.min()) / (strengths.max() - strengths.min() + 1)

    # Edge widths
    edges = list(H.edges())
    weights = np.array([H[u][v]["weight"] for u, v in edges]) if edges else np.array([])
    if weights.size:
        edge_widths = 1 + 20 * (weights - weights.min()) / (weights.max() - weights.min() + 1)
    else:
        edge_widths = []

    # Plot nodes
    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw_networkx_nodes(
        H, pos,
        node_size=node_sizes,
        node_color="#337aff",
        edgecolors="black",
        linewidths=1.5,
        ax=ax
    )

    # Plot weighted edges (if exist)
    if edges:
        nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.7, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(
        H, pos,
        font_size=12,
        font_weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        ax=ax
    )

    ax.set_title("Top 20 Fortune 500 Lobbying Affiliation Network", fontsize=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    # plt.show()


# CENTRALITY MEASURES
def compute_centralities(G):
    """Compute standard network centralities."""
    return {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "closeness": nx.closeness_centrality(G, distance="weight"),
        "eigenvector": nx.eigenvector_centrality(G, weight="weight"),
    }

def print_top_centralities(centralities, k=5):
    for name, values in centralities.items():
        print(f"\nTop {k} by {name.upper()} centrality:")
        for node, val in sorted(values.items(), key=lambda x: x[1], reverse=True)[:k]:
            print(f"{node}: {val:.4f}")

def main():
    df = pd.read_csv(DATA_PATH)
    print("Number of Bills:", df["bill_id"].nunique())

    company_bill_df = company_bill_edges(df)
    adjmat = build_adjacency_matrix(company_bill_df)
    # build_d3_graph(adjmat)

    # print("\nComputing thresholds and adjmat for PSNE input...")
    # thresholds = compute_thresholds(adjmat, percentile=75)
    # save_psne_input_with_threshold(adjmat, thresholds, PSNE_ADJMAT_TXT_PATH)

    print("\nBuilding graph...")
    G = build_networkx_graph(company_bill_df)
    H = extract_top_k_subgraph(G, k=20)

    print("\nPlotting network...")
    plot_affiliation_network(H, png_path=PNG_OUTPUT_PATH, gml_path=GML_OUTPUT_PATH)

    # print("\nComputing centralities...")
    # centralities = compute_centralities(G)
    # print_top_centralities(centralities)

if __name__ == "__main__":
    main()