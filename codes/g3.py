# 基础数据处理库
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
import pickle

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
import pymetis  # 图分割库

# 图嵌入和图神经网络相关库
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras
from common_functions import subGraphSAGE, graphsage_learning,load_graph_data

# -------------------------
# 全局参数
# -------------------------
FILE_PATH = "/home/sqp17/Projects/Two-level-GRL/dataset_WWW2019/"
EDGE_LIST_FILE = FILE_PATH + "dataset_WWW_friendship_new.txt"
NODE_FEATURES_FILE = "/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv"
DEFAULT_VAL = 1
COMMUNITY_THRESHOLD = 100
DIMENSIONS = 256

# -------------------------
# 辅助函数
# -------------------------


def perform_metis_partitioning(ig, k=50):
    """使用 METIS 进行图分割"""
    adjlist = [[int(node) for node in line.split()] for line in nx.generate_adjlist(ig.to_networkx())]
    nodes_dict = {node: idx for idx, node in enumerate(ig.vs["_nx_name"])}
    adjlist = [[nodes_dict[node] for node in neighbors] for neighbors in adjlist]
    partitioned = pymetis.part_graph(k, adjacency=adjlist)
    return partitioned[1]

def compute_node_embeddings(cd_algo, ig, node_features_encoded):
    """计算节点嵌入并更新节点特征"""
    node_embeddings = defaultdict(lambda: np.zeros(DIMENSIONS))
    start_time = time.time()

    for community_no, community in enumerate(cd_algo):
        if len(community) >= COMMUNITY_THRESHOLD:
            sub_node_embeddings = subGraphSAGE(ig, community, node_features_encoded)
            print(f"Community {community_no}: {len(sub_node_embeddings)} nodes embedded.")
            for idx, node_id in enumerate(community):
                node_embeddings[node_id] = sub_node_embeddings[idx]

    elapsed_time = time.time() - start_time
    print(f"Local GRL time: {elapsed_time:.2f} seconds")
    return node_embeddings

def create_reduced_graph(ig, cd_algo, node_features_encoded):
    """创建压缩后的全局图"""
    membership = cd_algo.membership
    minor_communities = {c[0] for c in Counter(membership).items() if c[1] < COMMUNITY_THRESHOLD}
    
    new_id = len(cd_algo)
    for idx, group in enumerate(membership):
        if group in minor_communities:
            membership[idx] = new_id
            new_id += 1

    idx_map = {old: new for new, old in enumerate(sorted(set(membership)))}
    membership = [idx_map[group] for group in membership]
    ig.contract_vertices(membership, combine_attrs="first")
    
    isin_filter = node_features_encoded["userID"].isin(ig.vs["_nx_name"])
    subgraph_features = node_features_encoded[isin_filter]
    subgraph_country_degree = subgraph_features[["countrycode_encoded", "degree"]].reset_index(drop=True)

    reduced_graph = StellarGraph.from_networkx(
        ig.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=subgraph_country_degree,
    )
    return reduced_graph

def merge_embeddings(node_embeddings, reduced_emb, cd_algo):
    """合并本地和全局嵌入"""
    j = 0
    for community in cd_algo:
        if len(community) < COMMUNITY_THRESHOLD:
            for node in community:
                node_embeddings[node] = reduced_emb[j]
                j += 1
        else:
            j += 1
    return node_embeddings

def perform_link_prediction(merged_graph):
    """执行链路预测任务"""
    start_time = time.time()
    edge_splitter_test = EdgeSplitter(merged_graph)
    graphsage_learning(edge_splitter_test, graph=merged_graph, epochs=10, need_dump=True)
    print(f"Time for global GRL: {time.time() - start_time:.2f} seconds")

# -------------------------
# 主流程
# -------------------------
if __name__ == "__main__":
    # 加载图数据和节点特征
    ig, user_graph, node_features_encoded = load_graph_data(EDGE_LIST_FILE, NODE_FEATURES_FILE)
    print(user_graph.info())

    # 社区检测
    LP = Graph.community_label_propagation(ig)
    print(f"Communities detected: {LP.summary()}")

    # 计算节点嵌入
    node_embeddings = compute_node_embeddings(LP, ig, node_features_encoded)

    # 创建压缩后的全局图
    reduced_graph = create_reduced_graph(ig, LP, node_features_encoded)
    print(reduced_graph.info())

    # 计算全局图嵌入
    start_time = time.time()
    edge_splitter_test = EdgeSplitter(reduced_graph)
    reduced_emb = graphsage_learning(edge_splitter_test, graph=reduced_graph, epochs=10, need_dump=True)
    print(f"Time for global GRL: {time.time() - start_time:.2f} seconds")

    # 合并嵌入
    node_embeddings = merge_embeddings(node_embeddings, reduced_emb, LP)

    # 创建最终的图并执行链路预测
    node_embeddings_df = pd.DataFrame(node_embeddings).transpose()
    merged_graph = StellarGraph.from_networkx(
        ig.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=node_embeddings_df,
    )
    perform_link_prediction(merged_graph)
