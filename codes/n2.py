# 基础数据处理库
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from collections import defaultdict
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

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter

# 可视化库
import matplotlib.pyplot as plt

# 全局参数
FILE_PATH = "/home/sqp17/Projects/Two-level-GRL/datasets/"
EDGE_LIST_FILE = FILE_PATH + "dataset_WWW_friendship_new.txt"
NODE_FEATURES_FILE = FILE_PATH + "node_features_encoded.csv"

P = 1.0
Q = 1.0
DIMENSIONS = 256
NUM_WALKS = 10
WALK_LENGTH = 80
WINDOW_SIZE = 10
NUM_ITER = 1
WORKERS = mp.cpu_count()

# -----------------------------------------
# Helper Functions
# -----------------------------------------


def process_communities(ig, node_features_encoded, pool_size):
    """处理所有社区并计算嵌入"""
    LP = ig.community_label_propagation()
    print(LP.summary())
    
    # 使用多进程计算社区嵌入
    with mp.Pool(pool_size) as pool:
        start = time.time()
        sub_node_embeddings_list = pool.map(
            lambda x: sub_Node2Vec(x, ig, node_features_encoded), list(LP)
        )
        elapsed_time = time.time() - start
    print(f"@@@ All Community Local GRL time (Pool Size: {pool_size}): {elapsed_time:.4f} seconds @@@")
    
    # 更新节点嵌入
    node_embeddings = defaultdict(lambda: np.zeros(DIMENSIONS))
    for community_idx, sub_node_embeddings in enumerate(sub_node_embeddings_list):
        for j, node_id in enumerate(LP[community_idx]):
            node_embeddings[node_id] = sub_node_embeddings[j]
    return node_embeddings, elapsed_time

def evaluate_link_prediction(node_embeddings, user_graph):
    """评估链路预测任务"""
    # Define edge splitter
    edge_splitter_test = EdgeSplitter(user_graph)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.1, method="global")
    
    edge_splitter_train = EdgeSplitter(graph_test, user_graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(p=0.1, method="global")
    
    examples_train, examples_val, labels_train, labels_val = train_test_split(
        examples, labels, train_size=0.75, test_size=0.25
    )
    
    # 运行链路预测
    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
    results = [
        run_link_prediction(op, node_embeddings, examples_train, labels_train, examples_val, labels_val)
        for op in binary_operators
    ]
    best_result = max(results, key=lambda r: r["score"])
    print(f"Best binary operator: {best_result['binary_operator'].__name__}")

    # 测试集链路预测
    for result in results:
        test_score = evaluate_link_prediction_model(
            result["classifier"], examples_test, labels_test, node_embeddings, result["binary_operator"]
        )
        print(f"Test ROC AUC using {result['binary_operator'].__name__}: {test_score:.4f}")
    return results, best_result

# -----------------------------------------
# Main Pipeline
# -----------------------------------------
if __name__ == "__main__":
    # 加载数据
    ig, user_graph, node_features_encoded = load_graph_dataEDGE_LIST_FILE, NODE_FEATURES_FILE)
    print(user_graph.info())
    
    # 处理社区并计算嵌入
    node_embeddings, local_grl_time = process_communities(ig, node_features_encoded, WORKERS)
    print(f"Number of embeddings: {len(node_embeddings)}")
    
    # 评估链路预测
    results, best_result = evaluate_link_prediction(node_embeddings, user_graph)
    print(f"Total Time: {local_grl_time:.2f} seconds")
