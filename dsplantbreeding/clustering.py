import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network

from IPython.core.display import display, HTML
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_correlation_network(df: pd.DataFrame, threshold: float = 0.9, colour_by=None, interactive=False, physics_options=False):
    """_summary_

    Args:
        df (pd.DataFrame): Gene expression dataframe
        threshold (float, optional): Cutoff above which to draw edge between nodes. Defaults to 0.9.
        colour_by (str, optional): 'degree', 'betweenness', or 'closeness'. Defaults to None.
    """
    colour_by_to_func = {'degree': nx.degree_centrality, 'betweenness': nx.betweenness_centrality, 'closeness': nx.closeness_centrality}
    # 1. Compute correlation matrix
    corr_matrix = df.T.corr(method="pearson") 

    # 2. Threshold correlations (you don’t want every edge, it’d be spaghetti)
    edges = [
        (gene1, gene2, corr_matrix.loc[gene1, gene2])
        for gene1 in corr_matrix.index
        for gene2 in corr_matrix.columns
        if gene1 < gene2 and corr_matrix.loc[gene1, gene2] > threshold
    ]

    # 3. Build network
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    if colour_by:
        node_color_metric = colour_by_to_func[colour_by](G)
        # Print top 10 genes by this metric
        top10 = sorted(node_color_metric.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 genes by centrality:")
        for gene, value in top10:
            print(f"{gene}: {value:.4f}")

        # Normalise colours for plotting
        values = [node_color_metric[node] for node in G.nodes()]
    else:
        values = "#1f78b4"

    if not interactive:
        # 4. Plot
        plt.figure(figsize=(10, 8))
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color=values, cmap=plt.cm.viridis)
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        # nx.draw_networkx_labels(G, pos, font_size=6)
        plt.title("Gene Correlation Network")
        plt.axis("off")
        plt.show()
    else:
        # Create pyvis network
        net = Network(notebook=True, cdn_resources='in_line', select_menu=True)
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        cmap = cm.get_cmap("viridis")
        # Add nodes with colormap
        for node, centrality in node_color_metric.items():
            rgba = cmap(norm(centrality))
            hex_color = mcolors.to_hex(rgba)
            net.add_node(
                node,
                title=f"{node}<br>Centrality: {centrality:.4f}",
                # value=centrality*100,  # adjust size if needed
                color=hex_color,
                size=100

            )

        # Add edges
        for u, v in G.edges():
            net.add_edge(u, v)

        if physics_options:
            net.show_buttons(filter_=['physics'])
        else:
            net.set_options(
            """
            {
            "physics": {
                "forceAtlas2Based": {
                "theta": 0.9,
                "gravitationalConstant": -290,
                "springLength": 5,
                "springConstant": 0.05,
                "damping": 0.7
                },
                "maxVelocity": 50,
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "timestep": 0.5
            }
            }
            """
            )
        
        net.show("nx.html")

        display(HTML('nx.html'))