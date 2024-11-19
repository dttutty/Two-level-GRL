# 基础数据处理库
import pandas as pd
import numpy as np
import pickle
import time
import multiprocessing as mp

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter

# 嵌入与机器学习库
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 可视化库
import matplotlib.pyplot as plt

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

# 全局参数
FILE_PATH = "/home/sqp17/Projects/Two-level-GRL/datasets/"
EDGE_LIST_FILE = FILE_PATH + "dataset_WWW_friendship_new.txt"
NODE_FEATURES_FILE = FILE_PATH + "node_features_encoded.csv"
COMMUNITY_INFO_FILE = FILE_PATH + "community_info.pickle"

P = 1.0
Q = 1.0
DIMENSIONS = 256
NUM_WALKS = 10
WALK_LENGTH = 80
WINDOW_SIZE = 10
NUM_ITER = 1
WORKERS = mp.cpu_count()
SIZE_THRESHOLD = 100

# -----------------------------------------
# Helper Functions
# -----------------------------------------

def generate_train_test_splits(graph, user_graph):
    """生成训练集与测试集"""
    edge_splitter_test = EdgeSplitter(user_graph)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.1, method="global")
    
    edge_splitter_train = EdgeSplitter(graph_test, user_graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(p=0.1, method="global")
    
    examples_train, examples_val, labels_train, labels_val = train_test_split(examples, labels, train_size=0.75, test_size=0.25)
    
    return graph_train, examples_train, examples_val, labels_train, labels_val, examples_test, labels_test

def train_global_embedding(graph_train):
    """计算全局 Node2Vec 嵌入"""
    start = time.time()
    embedding_train = node2vec_embedding(graph_train, "Train Graph")
    global_time = time.time() - start
    print(f"Global GRL time: {global_time:.2f} seconds")
    return embedding_train, global_time

def process_major_communities(ig, LP, node_features_encoded, embedding_train):
    """处理主要社区并更新嵌入"""
    start = time.time()
    for community_idx, community in enumerate(LP):
        if len(community) > SIZE_THRESHOLD:
            sub_node_embeddings = sub_Node2Vec(ig, node_features_encoded, community)
            print(f"Processing community {community_idx} with {len(community)} nodes.")
            for idx, node_id in enumerate(community):
                embedding_train[node_id] = sub_node_embeddings[idx]
    local_time = time.time() - start
    print(f"Major Community Local GRL time: {local_time:.2f} seconds")
    return embedding_train, local_time

def plot_pca_projection(link_features, labels_test):
    """PCA 降维并可视化嵌入"""
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(link_features)
    
    plt.figure(figsize=(12, 9))
    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=np.where(labels_test == 1, "b", "r"),
        alpha=0.5,
    )
    plt.title("PCA Projection of Link Features")
    plt.show()

# -----------------------------------------
# Main Pipeline
# -----------------------------------------
if __name__ == "__main__":
    # 加载数据
    ig, user_graph, node_features_encoded = load_graph_data(EDGE_LIST_FILE, NODE_FEATURES_FILE)
    print(user_graph.info())
    
    # 生成训练和测试集
    graph_train, examples_train, examples_val, labels_train, labels_val, examples_test, labels_test = generate_train_test_splits(user_graph, user_graph)
    
    # 计算全局嵌入
    embedding_train, global_time = train_global_embedding(graph_train)
    
    # 链路预测训练与验证
    binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
    results = [
        run_link_prediction(op, embedding_train, examples_train, labels_train, examples_val, labels_val)
        for op in binary_operators
    ]
    best_result = max(results, key=lambda r: r["score"])
    print(f"Best binary operator: {best_result['binary_operator'].__name__}")

    # 加载社区信息并处理主要社区
    with open(COMMUNITY_INFO_FILE, "rb") as f:
        LP = pickle.load(f)
    
    embedding_train, local_time = process_major_communities(ig, LP, node_features_encoded, embedding_train)
    
    # 测试集链路预测
    embedding_test = node2vec_embedding(graph_test, "Test Graph")
    for result in results:
        test_score = evaluate_link_prediction_model(
            result["classifier"],
            examples_test,
            labels_test,
            embedding_test,
            result["binary_operator"]
        )
        print(f"Test ROC AUC using {result['binary_operator'].__name__}: {test_score:.4f}")
    
    # 可视化 PCA 投影
    link_features = link_examples_to_features(examples_test, embedding_test, best_result["binary_operator"])
    plot_pca_projection(link_features, labels_test)
    
    print(f"Total Time: {global_time + local_time:.2f} seconds")
