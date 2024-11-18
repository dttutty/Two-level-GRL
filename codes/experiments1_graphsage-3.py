# 基础数据处理库
import pandas as pd
import numpy as np
import time
import pickle
from collections import Counter, defaultdict

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter, UnsupervisedSampler
import pymetis  # 图分割库

# 图嵌入和图神经网络相关库
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras

# 可视化库
import matplotlib.pyplot as plt

defaultVal = 1
incentiveVal = 1
penaltyVal = 1
# 극대화
# Edge between two user node (607,333 friendships)
file = '/home/sqp17/Projects/Two-level-GRL/dataset_WWW2019/dataset_WWW_friendship_new.txt'
G = nx.read_edgelist(file, nodetype=int, edgetype='Freindship')
iG = Graph.from_networkx(G)  # NetworkX to igraph
iG.vs["id"] = iG.vs["_nx_name"]
iG.es["weight"] = [defaultVal]*iG.ecount()
# METIS Partitioning !!!!
adjlist = []
for line in nx.generate_adjlist(G):
    adjlist.append([line])

for i in range(len(adjlist)):
    adjlist[i] = list(map(int, adjlist[i][0].split()))
# Adjusted index in adjacency list due to mismatch in indexing.
nodes = []
for el in adjlist:
    nodes.append(el[0])

nodes_dict = {k: v for v, k in enumerate(nodes)}
for i in range(len(adjlist)):
    for j in range(len(adjlist[i])):
        adjlist[i][j] = nodes_dict[adjlist[i][j]]
k = 50
partitioned = pymetis.part_graph(k, adjacency=adjlist)
membership = partitioned[1]
Counter(membership)
node_features_encoded = pd.read_csv(
    "/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv", index_col=0)
node_features_encoded
country_degree = pd.concat(
    [node_features_encoded['countrycode_encoded'], node_features_encoded['degree']], axis=1)
userGraph_country_deg = StellarGraph.from_networkx(iG.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=country_degree)
print(userGraph_country_deg.info())  # 전체 그래프 load


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
    history = model.fit(train_flow, epochs=10,
                        validation_data=test_flow, verbose=2)

    import stellargraph as sg
    sg.utils.plot_history(history)
    with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
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


'''# Baseline Experiment #
start = time.time()
edge_splitter_test = EdgeSplitter(userGraph_country_deg)
baseline_emb = graphsage_learning(edge_splitter_test,userGraph_country_deg)
print("time :", time.time() - start) '''
# ### 1. Community 단위로 RL 적용 (local RL) - 병렬 처리 가능
LP = Graph.community_label_propagation(iG)
cd_algo = LP
size_thresh = 100
minor_nodes_num = 0
major_comm_num = 0
for i in cd_algo:
    if len(i) <= size_thresh:
        minor_nodes_num += len(i)
    else:
        major_comm_num += 1
print("Communities with fewer than 100 nodes:", minor_nodes_num)
print("Number of communities with more than 100 nodes", major_comm_num)
print("Number of nodes in the reduced graph:", major_comm_num+minor_nodes_num)
# GraphSAGE Hyper-parameter Settings
batch_size = 20
epochs = 10
num_samples = [20, 10]
layer_sizes = [50, 50]


def subGraphSAGE(subgraphList):
    subgraph = iG.induced_subgraph(
        subgraphList, implementation="create_from_scratch")

    isin_filter = node_features_encoded['userID'].isin(subgraph.vs['id'])

    subgraph_features = node_features_encoded[isin_filter]
    subgraph_country_degree = pd.concat(
        [subgraph_features['countrycode_encoded'], subgraph_features['degree']], axis=1)
    subgraph_country_degree.reset_index(drop=True, inplace=True)

    subgraph_ = StellarGraph.from_networkx(subgraph.to_networkx(
    ), node_type_default="user", edge_type_default="friendship", node_features=subgraph_country_degree)

    subnodes = list(subgraph_.nodes())
    sub_unsupervised_samples = UnsupervisedSampler(
        subgraph_, nodes=subnodes, length=5, number_of_walks=1
    )

    sub_generator = GraphSAGELinkGenerator(subgraph_, batch_size, num_samples)
    sub_train_gen = sub_generator.flow(sub_unsupervised_samples)

    sub_graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=sub_generator, bias=True, dropout=0.0, normalize="l2"
    )

    x_inp, x_out = sub_graphsage.in_out_tensors()
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    sub_embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    sub_node_ids = subgraph_.nodes()
    sub_node_gen = GraphSAGENodeGenerator(
        subgraph_, batch_size, num_samples).flow(sub_node_ids)

    sub_node_embeddings = sub_embedding_model.predict(
        sub_node_gen, workers=4, verbose=1)

    return sub_node_embeddings


node_embeddings = defaultdict(lambda: np.zeros(dimensions))  # Initialization
start = time.time()
for community_no in range(len(cd_algo)):
    # Intra Community Embedding
    if len(cd_algo[community_no]) >= size_thresh:
        sub_node_embeddings = subGraphSAGE(cd_algo[community_no])
        print(len(sub_node_embeddings))

        # 전체 그래프에 대한 GraphSAGE에 의해 도출된 feature를 아예 덮어쓰는 것.
        j = 0
        for i in cd_algo[community_no]:
            node_embeddings[i] = sub_node_embeddings[j]
            j += 1
print("time(local) :", time.time() - start)
# ### Community는 하나의 node로 변환함으로써 축소된 Global graph 생성
# In[ ]:
membership = cd_algo.membership
counter = Counter(membership).most_common()
# In[ ]:
minor_vertexID = []
# community size 가 100보다 작으면 minor vertex ID list에 추가
for c in counter:
    if c[1] < size_thresh:
        minor_vertexID.append(c[0])
print(len(minor_vertexID))
# In[ ]:
# Minor community 인 애들은 노드 살려놔야하니까 membership 다시 부여하기
new_id = len(cd_algo)
for i in range(len(membership)):
    if membership[i] in minor_vertexID:
        membership[i] = new_id
        new_id += 1
# In[ ]:
idx_map = {}
n = 0
for i in sorted(dict(Counter(membership))):
    idx_map[i] = n
    n += 1
# In[ ]:
# 인덱스 순서대로 초기화
new_idx = 0
for i in range(len(membership)):
    membership[i] = idx_map[membership[i]]
# In[ ]:
# mean이나 median으로 하면 node id까지 반영되서 isin에서 노드 수가 달라져서 실행이 안됨..
iG.contract_vertices(membership, combine_attrs="first")
isin_filter = node_features_encoded['userID'].isin(iG.vs['_nx_name'])
subgraph_features = node_features_encoded[isin_filter]
subgraph_country_degree = pd.concat(
    [subgraph_features['countrycode_encoded'], subgraph_features['degree']], axis=1)
subgraph_country_degree.reset_index(drop=True, inplace=True)
len(subgraph_country_degree)
len(iG.to_networkx().nodes)
# ### RL 적용 (global RL) - 기존 graph에 비해 연산 크게 축소
reduced_globalG = StellarGraph.from_networkx(iG.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=subgraph_country_degree)
print(reduced_globalG.info())  # 전체 그래프 load
start = time.time()
edge_splitter_test = EdgeSplitter(reduced_globalG)
reudced_emb = graphsage_learning(edge_splitter_test, reduced_globalG)
print("Global GRL time :", time.time() - start)
# MERGE Final Results
j = 0
for community_no in range(len(cd_algo)):
    # Intra Community Embedding
    # Minor community 에 대한 노드들은 global GRL에서 결과 가져오기
    if len(cd_algo[community_no]) < size_thresh:
        for i in cd_algo[community_no]:
            node_embeddings[i] = reudced_emb[j]
            j += 1
    else:  # reduced node는 하나로 변환
        j += 1
j
len(node_embeddings)
'''
node_embeddings + reudced_emb = node features 만들어서
graph 생성 -> GraphSAGE Link prediction
* reudced_emb : 인덱스 보고 어떻게 전체 그래프 node feature랑 맞게 합칠지 고민해보기 
'''
# ### Downstream Task
node_embeddings_df = pd.DataFrame(node_embeddings).transpose()
node_embeddings_df
iG = Graph.from_networkx(G)  # NetworkX to igraph
iG.vs["id"] = iG.vs["_nx_name"]
iG.es["weight"] = [defaultVal]*iG.ecount()
merged_Graph = StellarGraph.from_networkx(iG.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=node_embeddings_df)
start = time.time()
edge_splitter_test = EdgeSplitter(merged_Graph)
graphsage_learning(edge_splitter_test, merged_Graph)
print("time(global) :", time.time() - start)
# In[ ]:
