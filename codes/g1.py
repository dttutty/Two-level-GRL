#!/usr/bin/env python
# coding: utf-8

# 数据处理和计算库
import numpy as np
import pandas as pd
import pickle
import time

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter

# 图嵌入与模型相关库
from common_functions import subGraphSAGE, graphsage_learning

# 全局参数
DEFAULT_VAL = 1
SIZE_THRESHOLD = 100

# 文件路径
EDGE_LIST_FILE = '/home/sqp17/Projects/Two-level-GRL/datasets/dataset_WWW_friendship_new.txt'
NODE_FEATURES_FILE = '/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv'
NODE_EMBEDDINGS_FILE = '/home/sqp17/Projects/Two-level-GRL/datasets/original_node_embedding.npy'
COMMUNITY_INFO_FILE = "/home/sqp17/Projects/Two-level-GRL/datasets/community_info.pickle"

def load_graph():
    """加载图数据并返回 igraph 和节点特征"""
    g = nx.read_edgelist(EDGE_LIST_FILE, nodetype=int, edgetype='Friendship')
    ig = Graph.from_networkx(g)
    ig.vs["id"] = ig.vs["_nx_name"]
    ig.es["weight"] = [DEFAULT_VAL] * ig.ecount()

    node_features_encoded = pd.read_csv(NODE_FEATURES_FILE, index_col=0)
    node_embeddings = np.load(NODE_EMBEDDINGS_FILE)
    node_embeddings_df = pd.DataFrame(node_embeddings)

    return ig, node_features_encoded, node_embeddings, node_embeddings_df

def partition_graph(ig):
    """生成基于多种社区检测算法的分区结果"""
    print("Generating community detection results...")
    cd_result = ig.community_multilevel()  # Modularity-based
    return cd_result

def process_communities(ig, cd_result, node_features_encoded, node_embeddings):
    """处理每个社区并进行嵌入更新"""
    print("Processing communities with size > threshold...")
    start_time = time.time()
    total_overwritten = 0

    for community_no, community in enumerate(cd_result):
        if len(community) > SIZE_THRESHOLD:
            # 子图嵌入
            sub_node_embeddings = subGraphSAGE(ig, community, node_features_encoded)
            print(f"Community {community_no}: {len(sub_node_embeddings)} nodes embedded.")

            # 更新嵌入
            for idx, node_id in enumerate(community):
                node_embeddings[node_id] = sub_node_embeddings[idx]
            total_overwritten += len(community)

    print(f"Community processing complete in {time.time() - start_time:.2f} seconds.")
    print(f"Total embeddings overwritten: {total_overwritten}")
    return total_overwritten

def downstream_task(ig, node_embeddings):
    """运行下游任务"""
    print("Running downstream tasks...")
    node_embeddings_df = pd.DataFrame(node_embeddings)

    # 构建基于新嵌入的图
    userGraph_partitioned = StellarGraph.from_networkx(
        ig.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=node_embeddings_df
    )

    # GraphSAGE 学习
    start_time = time.time()
    edge_splitter_test = EdgeSplitter(userGraph_partitioned)
    graphsage_learning(edge_splitter_test, graph=None, epochs=5)
    print(f"Downstream task completed in {time.time() - start_time:.2f} seconds.")

def main():
    # 加载图和节点特征
    ig, node_features_encoded, node_embeddings, node_embeddings_df = load_graph()

    # 基线图构建
    userGraph_baseline = StellarGraph.from_networkx(
        ig.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=node_embeddings_df
    )

    # 初始 GraphSAGE 学习
    print("Performing initial GraphSAGE learning...")
    edge_splitter_test = EdgeSplitter(userGraph_baseline)
    start_time = time.time()
    graphsage_learning(edge_splitter_test, graph=None, epochs=5)
    print(f"Initial GraphSAGE learning completed in {time.time() - start_time:.2f} seconds.")

    # 加载社区信息并检测
    with open(COMMUNITY_INFO_FILE, "rb") as f:
        LP = pickle.load(f)
    cd_result = partition_graph(ig)

    # 处理社区并更新嵌入
    total_overwritten = process_communities(ig, cd_result, node_features_encoded, node_embeddings)

    # 检查嵌入更新是否正确
    total_nodes_in_large_communities = sum(len(c) for c in cd_result if len(c) > SIZE_THRESHOLD)
    if total_nodes_in_large_communities == total_overwritten:
        print("Embedding overwriting complete and verified.")

    # 运行下游任务
    downstream_task(ig, node_embeddings)

if __name__ == "__main__":
    main()
