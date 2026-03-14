from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_circular(H, title, path):
    """
    Circular layout plot of subgraph H. Node size ∝ weighted degree.
    Parent directory is created automatically if it doesn't exist.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pos = nx.circular_layout(H)

    strengths = np.array([H.degree(n, weight="weight") for n in H.nodes()])
    sizes = (800 + 3000 * (strengths - strengths.min())
             / (strengths.max() - strengths.min() + 1))

    edges = list(H.edges())
    weights = np.array([H[u][v]["weight"] for u, v in edges]) if edges else np.array([])
    widths = (1 + 20 * (weights - weights.min())
              / (weights.max() - weights.min() + 1)) if weights.size else []

    fig, ax = plt.subplots(figsize=(16, 16))
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="#337aff",
                           edgecolors="black", linewidths=1.5, ax=ax)
    if edges:
        nx.draw_networkx_edges(H, pos, width=widths, alpha=0.7,
                               edge_color="gray", ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=12, font_weight="bold",
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.7),
                            ax=ax)

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {path}")
