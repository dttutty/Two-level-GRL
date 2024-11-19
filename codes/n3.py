#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
import multiprocessing as mp
from common_functions import (
    sub_Node2Vec, 
    node2vec_embedding, 
    run_link_prediction, 
    evaluate_link_prediction_model, 
    link_examples_to_features, 
    operator_hadamard, 
    operator_l1, 
    operator_l2, 
    operator_avg,
    load_graph_data
)

import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# -------------------------
# Global Parameters
# -------------------------
FILE_PATH = "/home/sqp17/Projects/Two-level-GRL/datasets/"
EDGE_LIST_FILE = FILE_PATH + "dataset_WWW_friendship_new.txt"
NODE_FEATURES_FILE = FILE_PATH + "node_features_encoded.csv"
DIMENSIONS = 256
SIZE_THRESHOLD = 100
WORKERS = mp.cpu_count()

# -------------------------
# Helper Functions
# -------------------------


def process_communities(ig, node_features_encoded, size_threshold):
    """Detect communities and classify as major or minor"""
    LP = ig.community_label_propagation()
    major_communities = [community for community in LP if len(community) >= size_threshold]
    minor_communities = [community for community in LP if len(community) < size_threshold]
    return LP, major_communities, minor_communities

def compute_major_embeddings(major_communities, ig, node_features_encoded):
    """Compute embeddings for major communities using parallel processing"""
    with mp.Pool(WORKERS) as pool:
        start = time.time()
        sub_node_embeddings_list = pool.map(
            lambda x: sub_Node2Vec(ig, node_features_encoded, x), major_communities
        )
        elapsed_time = time.time() - start
    print(f"Major Community Local GRL time: {elapsed_time:.4f} seconds")
    return sub_node_embeddings_list, elapsed_time

def update_node_embeddings(major_communities, sub_node_embeddings_list):
    """Update node embeddings with major community embeddings"""
    node_embeddings = defaultdict(lambda: np.zeros(DIMENSIONS))
    for community, embeddings in zip(major_communities, sub_node_embeddings_list):
        for idx, node_id in enumerate(community):
            node_embeddings[node_id] = embeddings[idx]
    return node_embeddings

def process_reduced_graph(ig, LP, minor_communities):
    """Create a reduced graph for minor communities"""
    membership = LP.membership
    minor_vertex_ids = [c[0] for c in Counter(membership).most_common() if c[1] < SIZE_THRESHOLD]
    
    new_id = len(LP)
    for i, group in enumerate(membership):
        if group in minor_vertex_ids:
            membership[i] = new_id
            new_id += 1
    
    idx_map = {old: new for new, old in enumerate(sorted(set(membership)))}
    membership = [idx_map[group] for group in membership]
    ig.contract_vertices(membership, combine_attrs="mean")
    
    reduced_graph = StellarGraph.from_networkx(
        ig.to_networkx(), 
        node_type_default="user", 
        edge_type_default="friendship"
    )
    return reduced_graph

def run_link_prediction_pipeline(node_embeddings, user_graph):
    """Perform link prediction and evaluate results"""
    edge_splitter_test = EdgeSplitter(user_graph)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.1, method="global")
    edge_splitter_train = EdgeSplitter(graph_test, user_graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(p=0.1, method="global")
    
    examples_train, examples_val, labels_train, labels_val = train_test_split(
        examples, labels, train_size=0.75, test_size=0.25
    )
    
    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
    results = [
        run_link_prediction(op, node_embeddings, examples_train, labels_train, examples_val, labels_val)
        for op in binary_operators
    ]
    
    best_result = max(results, key=lambda r: r["score"])
    print(f"Best binary operator: {best_result['binary_operator'].__name__}")
    
    # Evaluate on test set
    for result in results:
        test_score = evaluate_link_prediction_model(
            result["classifier"], examples_test, labels_test, node_embeddings, result["binary_operator"]
        )
        print(f"Test ROC AUC using {result['binary_operator'].__name__}: {test_score:.4f}")
    
    return results, best_result

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    ig, user_graph, node_features_encoded = load_graph_data(EDGE_LIST_FILE, NODE_FEATURES_FILE)
    print(user_graph.info())
    
    # Process communities
    LP, major_communities, minor_communities = process_communities(ig, node_features_encoded, SIZE_THRESHOLD)
    print(f"Number of major communities: {len(major_communities)}")
    print(f"Number of minor nodes: {sum(len(c) for c in minor_communities)}")
    
    # Compute embeddings for major communities
    major_embeddings, local_grl_time = compute_major_embeddings(major_communities, ig, node_features_encoded)
    node_embeddings = update_node_embeddings(major_communities, major_embeddings)
    
    # Process reduced graph
    reduced_graph = process_reduced_graph(ig, LP, minor_communities)
    print(reduced_graph.info())
    
    # Compute embeddings for reduced graph
    start = time.time()
    reduced_global_embeddings = node2vec_embedding(reduced_graph, "Reduced Global Graph")
    global_grl_time = time.time() - start
    print(f"Global reduced GRL time: {global_grl_time:.4f} seconds")
    
    # Combine embeddings
    for community in minor_communities:
        for node_id in community:
            node_embeddings[node_id] = reduced_global_embeddings[node_id]
    
    # Run link prediction pipeline
    run_link_prediction_pipeline(node_embeddings, user_graph)
    
    print(f"Total Time: {local_grl_time + global_grl_time:.2f} seconds")
