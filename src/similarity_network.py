'''
Prathit Kurup, Victoria Figueroa
02/18/2026

Build a company-to-company similarity network based on shared lobbying activity.
Produce adjacency matrix (and threshold calculations for PSNE), 
NetworkX graph, and centrality measures.
'''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from d3graph import d3graph, vec2adjmat
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "../data/fortune500_lda_reports.csv"
EDGE_OUTPUT_PATH = "fortune500_bills_similarity.csv"
ADJMAT_OUTPUT_PATH = "fortune500_similarity_adjmat.csv"
PSNE_ADJMAT_TXT_PATH = "./influence_game_psne_calculation/fortune500_similarity_psne_game_input.txt"
GML_OUTPUT_PATH = "../visualizations/fortune_500_lobbying_similarity_network.gml"
PNG_OUTPUT_PATH = "../visualizations/fortune_500_lobbying_similarity_network.png"

# CREATE SIMILARITY NETWORK AND ADJACENCY MATRIX
def company_bill_edges(dataset_df, edge_list_path=EDGE_OUTPUT_PATH):
    """
    Create company similarity edges using cosine similarity
    over company spending vectors across bills.
    """

    '''
    FRACTIONAL CO-INVESTMENT 
    '''

    # ---- 1. Compute total spend per bill ----
    bill_totals = (
        dataset_df
        .groupby("bill_id")["amount"]
        .sum()
        .reset_index(name="total_bill_amount")
    )

    # ---- 2. Merge totals back into main df ----
    df = dataset_df.merge(bill_totals, on="bill_id", how="left")

    # ---- 3. Compute fractional contribution per company per bill ----
    df["fractional_amount"] = df["amount"] / df["total_bill_amount"]

    # ---- 4. Group companies per bill ----
    bill_company_df = (
        df.groupby("bill_id")
          .apply(lambda x: list(zip(x["client_name"], x["fractional_amount"])))
          .reset_index(name="companies")
    )

    # ---- 5. Build weighted edges ----
    edge_records = []

    for _, row in bill_company_df.iterrows():
        bill_id = row["bill_id"]
        companies = row["companies"]   # [(company, frac_amount), ...]

        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                (c1, f1) = companies[i]
                (c2, f2) = companies[j]

                if c1 != c2:
                    # fractional co-investment weight
                    # pair_weight = min(f1, f2)

                    # edge_records.append({
                    #     "source": c1,
                    #     "target": c2,
                    #     "bill_id": bill_id,
                    #     "weight": pair_weight
                    # })

                    bc_similarity = 1 - abs(f1 - f2) / (f1 + f2)
                    edge_records.append({
                        "source": c1,
                        "target": c2,
                        "bill_id": bill_id,
                        "weight": bc_similarity
                    })

    # ---- 6. Aggregate across bills ----
    edges_df = pd.DataFrame(edge_records)

    aggregate_edges = (
        edges_df
        .groupby(["source", "target"])["weight"]
        .sum()
        .reset_index()
    )

    # ---- 7. Clean + save ----
    aggregate_edges["weight"] = pd.to_numeric(aggregate_edges["weight"], errors="coerce")
    aggregate_edges = aggregate_edges[aggregate_edges["weight"] > 0]
    aggregate_edges.to_csv(edge_list_path, index=False)
    # print(aggregate_edges["weight"].describe())

    return aggregate_edges

    '''
    COSINE SIMILARITY
    '''
    # # Build company × bill spending matrix
    # company_bill_matrix = (
    #     dataset_df
    #     .groupby(["client_name", "bill_id"])["amount"]
    #     .sum()
    #     .unstack(fill_value=0)
    # )

    # companies = company_bill_matrix.index.tolist()
    # spending_vectors = company_bill_matrix.values

    # # Compute cosine similarity matrix
    # sim_matrix = cosine_similarity(spending_vectors)

    # edge_records = []

    # for i in range(len(companies)):
    #     for j in range(i + 1, len(companies)):
    #         sim = sim_matrix[i, j]
    #         if sim > 0:
    #             edge_records.append({
    #                 "source": companies[i],
    #                 "target": companies[j],
    #                 "weight": sim
    #             })

    # edges_df = pd.DataFrame(edge_records)
    # edges_df.to_csv(edge_list_path, index=False)

    # return edges_df

    '''
    PER BILL SIMILARITY
    '''
    # # For each bill, collect all companies that lobbied on it
    # bill_company_df = dataset_df.groupby("bill_id")[["client_name", "amount"]] \
    #     .apply(lambda x: list(zip(x["client_name"], x["amount"]))) \
    #     .reset_index(name="company_amounts")

    # # Generate edges for each bill's company list (complete subgraph)
    # edge_records = []
    # for _, row in bill_company_df.iterrows():
    #     pairs = row["company_amounts"]
    #     bill_id = row["bill_id"]

    #     for i in range(len(pairs)):
    #         for j in range(i + 1, len(pairs)):
    #             c1, a1 = pairs[i]
    #             c2, a2 = pairs[j]
    #             if c1 != c2 and (a1 + a2) > 0:
    #                 similarity = (2 * min(a1, a2)) / (a1 + a2)   # per-bill similarity score
    #                 edge_records.append({
    #                     "source": c1,
    #                     "target": c2,
    #                     "bill_id": bill_id,
    #                     "similarity": similarity
    #                 })
    
    # # Aggregate shared bills into weighted edges, where weight = number of shared bills
    # edges_df = pd.DataFrame(edge_records)
    # bill_counts = dataset_df.groupby("client_name")["bill_id"].nunique().to_dict()

    # # aggregate similarity
    # agg = edges_df.groupby(["source", "target"]).agg(
    #     mean_similarity=("similarity", "mean"),
    #     shared_bills=("bill_id", "nunique")
    # ).reset_index()

    # # overlap factor
    # agg["overlap"] = agg.apply(
    #     lambda r: r["shared_bills"] / min(bill_counts.get(r["source"], 1), bill_counts.get(r["target"], 1)),
    #     axis=1
    # )
    
    # # final weighted similarity
    # agg["weight"] = agg["mean_similarity"] * agg["overlap"]
    # aggregate_edges = agg[["source", "target", "weight"]]
    # aggregate_edges = aggregate_edges[aggregate_edges["weight"] > 0]
    # aggregate_edges.to_csv(edge_list_path, index=False)
    # return aggregate_edges

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
def plot_similarity_network(H, png_path=PNG_OUTPUT_PATH, gml_path=GML_OUTPUT_PATH):
    """Visualize top-k similarity network."""
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

    ax.set_title("Top 20 Fortune 500 Lobbying Similarity Network", fontsize=18, fontweight="bold")
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

def print_top_centralities(centralities, k=10):
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
    plot_similarity_network(H, png_path=PNG_OUTPUT_PATH, gml_path=GML_OUTPUT_PATH)

    # print("\nComputing centralities...")
    # centralities = compute_centralities(G)
    # print_top_centralities(centralities, k=10)

if __name__ == "__main__":
    main()