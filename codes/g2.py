#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
from collections import defaultdict
import multiprocessing as mp
from common_functions import subGraphSAGE, graphsage_learning
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
import matplotlib.pyplot as plt

# -------------------------
# Global Parameters
# -------------------------
FILE_PATH = "/home/sqp17/Projects/Two-level-GRL/datasets/"
EDGE_LIST_FILE = FILE_PATH + "dataset_WWW_friendship_new.txt"
NODE_FEATURES_FILE = FILE_PATH + "node_features_encoded.csv"
DIMENSIONS = 256
WORKERS = mp.cpu_count()
COMMUNITY_THRESHOLD = 100

# -------------------------
# Helper Functions
# -------------------------
def load_graph_data():
    """Load graph data and node features"""
    edges = np.loadtxt(EDGE_LIST_FILE, dtype=int)
    node_features_encoded = pd.read_csv(NODE_FEATURES_FILE, index_col="userID")
    country_degree = node_features_encoded[["countrycode_encoded", "degree"]]
    edges_array = np.c_[edges, np.ones(len(edges))]

    user_graph = StellarGraph(
        nodes=pd.DataFrame(country_degree),
        edges=pd.DataFrame(edges_array, columns=["source", "target", "weight"]),
        node_type_default="user",
        edge_type_default="friendship",
    )

    g = nx.read_edgelist(EDGE_LIST_FILE, nodetype=int, edgetype="Friendship")
    ig = Graph.from_networkx(g)
    ig.vs["id"] = ig.vs["_nx_name"]
    ig.es["weight"] = [1.0] * ig.ecount()
    return ig, user_graph, node_features_encoded

def perform_community_detection(ig):
    """Perform community detection using label propagation"""
    LP = ig.community_label_propagation()
    print("Community Detection Summary:")
    LP.summary()
    return LP

def update_embeddings_with_communities(ig, LP, node_features_encoded):
    """Compute and update node embeddings for all communities"""
    node_embeddings = defaultdict(lambda: np.zeros(DIMENSIONS))
    start_time = time.time()

    for community_no, community in enumerate(LP):
        # Compute embeddings for each community
        sub_node_embeddings = subGraphSAGE(ig, community, node_features_encoded, verbose=0, dropout=0.3)
        print(f"Community {community_no}: {len(sub_node_embeddings)} nodes embedded.")
        for j, node_id in enumerate(community):
            node_embeddings[node_id] = sub_node_embeddings[j]

    elapsed_time = time.time() - start_time
    print(f"Time taken for community-based embedding: {elapsed_time:.2f} seconds")
    return node_embeddings

def perform_link_prediction(merged_graph, epochs=50):
    """Perform link prediction using GraphSAGE"""
    start_time = time.time()
    edge_splitter_test = EdgeSplitter(merged_graph)
    graphsage_learning(edge_splitter_test, graph=merged_graph, epochs=epochs)
    elapsed_time = time.time() - start_time
    print(f"Time for link prediction: {elapsed_time:.2f} seconds")

# -------------------------
# Main Pipeline
# -------------------------
if __name__ == "__main__":
    # Step 1: Load graph data and node features
    ig, user_graph, node_features_encoded = load_graph_data()
    print(user_graph.info())

    # Step 2: Perform community detection
    LP = perform_community_detection(ig)

    # Step 3: Compute community embeddings and update the graph
    node_embeddings = update_embeddings_with_communities(ig, LP, node_features_encoded)

    # Step 4: Create a new StellarGraph with updated embeddings
    node_embeddings_df = pd.DataFrame(node_embeddings).transpose()
    merged_graph = StellarGraph.from_networkx(
        ig.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=node_embeddings_df,
    )

    # Step 5: Perform link prediction on the merged graph
    perform_link_prediction(merged_graph)

# -------------------------
# Optimized Functions
# -------------------------
def subGraphSAGE(ig, subgraph_list, node_features_encoded, verbose=1, dropout=0.3, batch_size=20, num_samples=[20, 10], layer_sizes=[50, 50]):
    """Compute GraphSAGE embeddings for a subgraph"""
    subgraph = ig.induced_subgraph(subgraph_list, implementation="create_from_scratch")
    isin_filter = node_features_encoded.index.isin(subgraph.vs["id"])
    subgraph_features = node_features_encoded[isin_filter]
    subgraph_features.reset_index(drop=True, inplace=True)
    
    subgraph_stellar = StellarGraph.from_networkx(
        subgraph.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=subgraph_features,
    )

    # GraphSAGE Model Setup
    generator = GraphSAGENodeGenerator(subgraph_stellar, batch_size, num_samples)
    train_flow = generator.flow(subgraph_stellar.nodes())

    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=dropout
    )
    x_inp, x_out = graphsage.in_out_tensors()
    embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
    node_embeddings = embedding_model.predict(train_flow, workers=WORKERS, verbose=verbose)
    return node_embeddings
