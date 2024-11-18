# 2) Community-only: Perform Local GRL on all communities without global GRL on the entire graph (applicable to both major and minor communities).
#
# => Minimum accuracy, parallel processing possible (optimal processing speed).
# * Processing speed = Measure the time taken to perform Parallel Local GRL
# 基础数据处理库
import pandas as pd
import numpy as np
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from stellargraph import StellarGraph

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter, UnsupervisedSampler

# 图嵌入和图神经网络相关库
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras

# 可视化库
import matplotlib.pyplot as plt




# Edge between two user node (607,333 friendships)
file = '/home/sqp17/Projects/Two-level-GRL/datasets/dataset_WWW_friendship_new.txt'
G = nx.read_edgelist(file, nodetype=int, edgetype='Freindship')


# Step 1: 读取边列表
edges = np.loadtxt("/home/sqp17/Projects/Two-level-GRL/datasets/dataset_WWW_friendship_new.txt", dtype=int)

# Step 2: 读取节点特征并设置索引
node_features_encoded = pd.read_csv(
    "/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv", index_col="userID"
)
# 只保留需要的 'countrycode_encoded' 和 'degree' 列
country_degree = node_features_encoded[["countrycode_encoded", "degree"]]

# Step 3: 将边列表转换为边特征数据
# 所有边权重均设为1.0
edges_array = np.c_[edges, np.ones(len(edges))]

# Step 4: 构建 StellarGraph 对象
# 节点的 feature dataframe 为 `country_degree`，边的 source, target, weight 从 edges_array 中获取
userGraph_country_deg = StellarGraph(
    nodes=pd.DataFrame(country_degree), 
    edges=pd.DataFrame(edges_array, columns=["source", "target", "weight"]),
    node_type_default="user",
    edge_type_default="friendship"
)

# Step 5: 打印图信息
print(userGraph_country_deg.info())


iG = Graph.from_networkx(G)  # NetworkX to igraph
iG.vs["id"] = iG.vs["_nx_name"]  # 将节点的名称属性设置为节点ID
iG.es["weight"] = [1.0]*iG.ecount()  # 为每条边赋予权重，所有边权重均为1.0

node_features_encoded = pd.read_csv(
    "/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv", index_col=0)
country_degree = pd.concat(
    [node_features_encoded['countrycode_encoded'], node_features_encoded['degree']], axis=1)



# 使用标签传播算法进行社区检测
LP = Graph.community_label_propagation(iG)  # 커뮤니티 디텍션
LP.summary()  # 输出标签传播算法的社区检测结果摘要

# mu = Graph.community_multilevel(ig)
# mu.summary()

eigen = Graph.community_multilevel(iG)  # 4 Eigenvector-based => OK
eigen.summary()

# GraphSAGE Hyper-parameter Settings
batch_size = 20
epochs = 5
num_samples = [20, 10]
layer_sizes = [50, 50]


def subGraphSAGE(subgraphList):
    # 从提供的节点列表创建一个诱导子图
    subgraph = iG.induced_subgraph(
        subgraphList, implementation="create_from_scratch")

    # 筛选出属于子图的节点特征
    isin_filter = node_features_encoded['userID'].isin(subgraph.vs['id'])
    subgraph_features = node_features_encoded[isin_filter]

    # 提取国家编码和度数信息，并重置索引
    subgraph_country_degree = pd.concat(
        [subgraph_features['countrycode_encoded'], subgraph_features['degree']], axis=1)
    subgraph_country_degree.reset_index(drop=True, inplace=True)

    # 将子图从 iGraph 格式转换为 StellarGraph 格式，并添加节点特征
    subgraph_ = StellarGraph.from_networkx(
        subgraph.to_networkx(), node_type_default="user", edge_type_default="friendship", node_features=subgraph_country_degree
    )

    # 如果子图中节点数量超过 100，则打印节点数量
    if len(subgraph_.nodes()) > 100:
        print("Node数量: ", len(subgraph_.nodes()))

    # 获取子图中的节点列表
    subnodes = list(subgraph_.nodes())

    # 创建无监督采样器，用于生成随机游走序列
    sub_unsupervised_samples = UnsupervisedSampler(
        subgraph_, nodes=subnodes, length=5, number_of_walks=1
    )

    # 创建 GraphSAGE 的数据生成器，用于链路预测任务
    sub_generator = GraphSAGELinkGenerator(subgraph_, batch_size, num_samples)
    sub_train_gen = sub_generator.flow(sub_unsupervised_samples)

    # 定义 GraphSAGE 模型，包含神经网络层结构和超参数
    sub_graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=sub_generator, bias=True, dropout=0.3, normalize="l2"
    )

    # 获取模型输入和输出张量
    x_inp, x_out = sub_graphsage.in_out_tensors()
    x_inp_src = x_inp[0::2]  # 源节点输入
    x_out_src = x_out[0]     # 源节点输出嵌入
    sub_embedding_model = keras.Model(
        inputs=x_inp_src, outputs=x_out_src)  # 构建嵌入模型

    # 创建节点生成器，用于生成子图节点的嵌入
    sub_node_ids = subgraph_.nodes()
    sub_node_gen = GraphSAGENodeGenerator(
        subgraph_, batch_size, num_samples).flow(sub_node_ids)

    # 预测子图中节点的嵌入
    sub_node_embeddings = sub_embedding_model.predict(
        sub_node_gen, workers=4, verbose=0)

    return sub_node_embeddings  # 返回子图的节点嵌入


node_embeddings = defaultdict(lambda: np.zeros(dimensions))  # Initialization

start = time.time()
for community_no in range(len(LP)):
    # Intra Community Embedding
    sub_node_embeddings = subGraphSAGE(LP[community_no])
    print(len(sub_node_embeddings))
    # 전체 그래프에 대한 GraphSAGE에 의해 도출된 feature를 아예 덮어쓰는 것.
    j = 0
    for i in LP[community_no]:
        node_embeddings[i] = sub_node_embeddings[j]
        j += 1
print("time :", time.time() - start)

len(node_embeddings)
# In[ ]:
"""start = time.time()
total_result = 0
pool = ProcessPoolExecutor()
procs = []
for community_no in range(len(LP)):
    procs.append(pool.submit(subGraphSAGE, LP[community_no]))
end = time.time()
print("수행시간: %f 초" % (end - start))
print("총결괏값: %s" % total_result)"""
# In[ ]:
"""pool_obj = mp.Pool(5)
start = time.time()
community_idx = 0
for ret in pool_obj.imap(subGraphSAGE,list(LP)):
    sub_node_embeddings = ret
    j=0
    for i in LP[community_idx]:
        node_embeddings[i] = sub_node_embeddings[j]
        j += 1
    community_idx += 1
    
len(node_embeddings)
for community_idx in range(len(LP)):
    sub_node_embeddings = sub_node_embeddings_list[community_idx]
    j=0
    for i in LP[community_idx]:
        node_embeddings[i] = sub_node_embeddings[j]
        j += 1
len(node_embeddings)
# 왜 안될 까 !!!!!!!!!!!!!!!!!!!!
"""
# ### 전체 그래프에 대해서 Link Prediction 결과 확인


def graphsage_learning(edge_splitter_test, graph):
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
    # reduced graph G_test with the sampled links removed:
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )
    # Define an edge splitter on the reduced graph G_test:
    edge_splitter_train = EdgeSplitter(G_test)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
    # reduced graph G_train with the sampled links removed:
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )
    train_gen = GraphSAGELinkGenerator(
        G_train, batch_size, num_samples, weighted=True)
    train_flow = train_gen.flow(
        edge_ids_train, edge_labels_train, shuffle=True)
    test_gen = GraphSAGELinkGenerator(
        G_test, batch_size, num_samples, weighted=True)
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
    )
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=["acc"],
    )
    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)
    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    print()
    print("#################################################################################################################")
    history = model.fit(train_flow, epochs=50,
                        validation_data=test_flow, verbose=2)

    import stellargraph as sg
    sg.utils.plot_history(history)
    print()
    print("################################################################################################################")
    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)
    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_ids = graph.nodes()
    node_gen = GraphSAGENodeGenerator(
        graph, batch_size, num_samples).flow(node_ids)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

    return node_embeddings


node_embeddings_df = pd.DataFrame(node_embeddings).transpose()

node_embeddings_df

merged_Graph = StellarGraph.from_networkx(iG.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=node_embeddings_df)

start = time.time()
edge_splitter_test = EdgeSplitter(merged_Graph)
graphsage_learning(edge_splitter_test, merged_Graph)
print("time for link prediction :", time.time() - start)
# In[ ]:
"""
@@@ NO PARALLEL PROCESSING VERSION @@@
start = time.time()
for community_no in range(len(cd_algo)):
    # Intra Community Embedding 
    if len(cd_algo[community_no]) >= 100:
        sub_node_embeddings = subGraphSAGE(cd_algo[community_no])
        print(len(sub_node_embeddings))
        
        # 전체 그래프에 대한 GraphSAGE에 의해 도출된 feature를 아예 덮어쓰는 것. 
        j=0
        for i in cd_algo[community_no]:
            node_embeddings[i] = sub_node_embeddings[j]
            j += 1
print("time :", time.time() - start) """
