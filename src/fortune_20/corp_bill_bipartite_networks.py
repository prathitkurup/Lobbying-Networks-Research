# Prathit Kurup, Victoria Figueroa
# December 2025
# This script uses the LobbyView data we scraped to create a network using NetworkX of Fortune 100 corporations and the amount they spent on each bill in 2019 (116th Congress)

import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from networkx.algorithms import bipartite


def load_and_aggregate(csv_path):
    # We want to vizualize just the top bills of 2019
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

    # Load lobbying data and aggregate total spending per (client_name, bill_id).
    edge_weights = {}
    companies = set()
    bills = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # lob_id = row["lob_id"]
            bill_id = row["bill_id"]
            if bill_id not in top_bills:
                continue
            client_name = row["client_name"]
            report_amount = float(row["amount"]) if row["amount"] else 0.0
            key = (client_name, bill_id)
            edge_weights[key] = edge_weights.get(key, 0.0) + report_amount
            companies.add(client_name)
        bills.update(bill_id for bill_id in top_bills)
        # update the edge weights to include bills with zero spending from some companies
        for bill_id in top_bills:
            for company in companies:
                key = (company, bill_id)
                if key not in edge_weights:
                    edge_weights[key] = 0.0
    
    # Print graph statistics
    print(f"Number of companies: {len(companies)}")
    print(f"Number of bills: {len(bills)}")
    print(f"Total lobbying dollars spent: {sum(edge_weights.values())}")

    return edge_weights, companies, bills


def build_bipartite_graph(edge_weights):
    # Use aggregated dollar amounts spent to create edges in bipartite graph
    B = nx.DiGraph()

    for (client_name, bill_id), total_amount_spent in edge_weights.items():
        B.add_node(client_name, bipartite=0, node_type="corporation")
        B.add_node(bill_id, bipartite=1, node_type="bill")
        B.add_edge(client_name, bill_id, weight=total_amount_spent)

    labels = {}
    for node, data in B.nodes(data=True):
        if data["node_type"] == "bill":
            labels[node] = node.replace("-116", "")
        else:
            labels[node] = node
    
    # Save the graph
    nx.write_graphml(B, "top_10_tech_bill_network.graphml")
    print("Graph saved to top_10_tech_bill_network.graphml")

    return B, labels


# def create_node_labels(B, companies):
#     # For plotting in networkX, we want to create labels for each node
#     labels = {}
#     for node, data in B.nodes(data=True):
#         if data["bipartite"] == 0:
#             labels[node] = node if node in companies else str(node)
#         else:
#             labels[node] = node
#     return labels


def visualize_graph(B, labels=None, bills=None):
    print("\nGenerating visualization...")

    # Visualize bipartite graph with clear left-right separation.
    left_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]      # Corporations
    # right_nodes = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]     # Bills

    pos = nx.bipartite_layout(B, left_nodes, align="vertical")
    weights = [B[u][v]["weight"] for u, v in B.edges()]
    # widths = np.log1p(weights)

    plt.figure(figsize=(14, 10))

    # Example node sizes
    node_sizes = {n: 300 + 50 * B.degree(n) for n in B.nodes()}

# Convert node_size → font_size
    label_sizes = {
        n: max(6, math.sqrt(size) / 2)
        for n, size in node_sizes.items()
    }

    # Draw the corporations and bills with different colors
    nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_color="blue", alpha = 0.2, node_size=1500, label="Corporations" )
    nx.draw_networkx_nodes(B, pos, nodelist=bills, node_color="red", alpha = 0.2, node_size=1500, label="Bills")
    # Only draw edges with non-zero weight
    non_zero_edges = [(u, v) for u, v in B.edges() if B[u][v]["weight"] > 0]
    nx.draw_networkx_edges(B, pos, edgelist=non_zero_edges, width=1, edge_color="gray")
    nx.draw_networkx_labels(B, pos, labels=labels, font_size=11, font_weight="bold", font_color="black")
    plt.title("Fortune 20 Companies Lobbying for Top 10 Tech Bills in 116th Congress", fontsize=18, fontweight='bold')
    # plt.legend()
    plt.axis("off")
    plt.tight_layout()

    # Save the visualization
    plt.savefig("top_10_tech_bills.png", dpi=300, bbox_inches='tight')
    print("Visualization saved as  top_10_tech_bills.png")

    # Show the plot
    plt.show()

def main():
    csv_path = "fortune100_client_lob_bill_amount.csv"

    edge_weights, companies, bills = load_and_aggregate(csv_path)
    B, labels = build_bipartite_graph(edge_weights)
    visualize_graph(B, labels, bills)

if __name__ == "__main__":
    main()
