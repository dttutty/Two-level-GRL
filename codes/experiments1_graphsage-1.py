#!/usr/bin/env python
# coding: utf-8
# 数据处理和计算库
import numpy as np
import pandas as pd
import pickle
import sys
import time

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter, UnsupervisedSampler

# 图嵌入与模型相关库
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras

# 可视化库
import matplotlib.pyplot as plt

defaultVal = 1
incentiveVal = 1
penaltyVal = 1

# Maximization
# Edge between two user node (607,333 friendships)
file = '/home/sqp17/Projects/Two-level-GRL/datasets/dataset_WWW_friendship_new.txt'
g = nx.read_edgelist(file, nodetype=int, edgetype='Freindship')
ig = Graph.from_networkx(g)  # NetworkX to igraph
ig.vs["id"] = ig.vs["_nx_name"]
ig.es["weight"] = [defaultVal]*ig.ecount()
node_features_encoded = pd.read_csv(
    "/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv", index_col=0)
node_features_encoded
country_degree = pd.concat(
    [node_features_encoded['countrycode_encoded'], node_features_encoded['degree']], axis=1)
node_embeddings = np.load(
    '/home/sqp17/Projects/Two-level-GRL/datasets/original_node_embedding.npy')
node_embeddings_df = pd.DataFrame(node_embeddings)

userGraph_baseline = StellarGraph.from_networkx(ig.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=node_embeddings_df)


def graphsage_learning(edge_splitter_test):
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
    batch_size = 20
    epochs = 5
    num_samples = [20, 10]
    train_gen = GraphSAGELinkGenerator(
        G_train, batch_size, num_samples, weighted=True)
    train_flow = train_gen.flow(
        edge_ids_train, edge_labels_train, shuffle=True)
    test_gen = GraphSAGELinkGenerator(
        G_test, batch_size, num_samples, weighted=True)
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
    layer_sizes = [50, 50]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
    )
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)
    from tensorflow import keras
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
    history = model.fit(train_flow, epochs=epochs,
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


start = time.time()
edge_splitter_test = EdgeSplitter(userGraph_baseline)
graphsage_learning(edge_splitter_test)
print("time :", time.time() - start)
# ### 이까지는 전체 그래프 (114,324개) Node embedding한 결과.
with open("/home/sqp17/Projects/Two-level-GRL/datasets/community_info.pickle", "rb") as f:
    LP = pickle.load(f)  # 1 Label-propagation-based
mu = Graph.community_multilevel(ig)  # 2 Modularity-based => OK
mu.summary()
eigen = Graph.community_leading_eigenvector(ig)  # 4 Eigenvector-based => OK
eigen.summary()
random_walk = Graph.community_walktrap(ig)  # 5 Random walks-based => OK
random_walk.summary()
random_walk = random_walk.as_clustering()
random_walk.summary()
info = Graph.community_infomap(ig)  # 7 InfoMAP algorithm => OK
info.summary()
# 알고리즘 선택 @@@
cd_result = mu
# In[ ]:
size_threshold = 100
batch_size = 20
epochs = 5
num_samples = [20, 10]
layer_sizes = [50, 50]


def subGraphSAGE(subgraphList):
    subgraph = ig.induced_subgraph(
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


start = time.time()
count = 0
for community_no in range(len(cd_result)):

    # Intra Community Embedding
    if len(cd_result[community_no]) > size_threshold:
        sub_node_embeddings = subGraphSAGE(cd_result[community_no])
        print(len(sub_node_embeddings))

        # 전체 그래프에 대한 GraphSAGE에 의해 도출된 feature를 아예 덮어쓰는 것.
        j = 0
        for i in cd_result[community_no]:
            node_embeddings[i] = sub_node_embeddings[j]
            j += 1
            count += 1  # 총 덮어쓴 횟수 (=100보다 큰 커뮤니티에 포함된 노드의 개수)
print("time :", time.time() - start)
c = 0
for s in range(len(cd_result)):
    if len(cd_result[s]) > 100:
        c += len(cd_result[s])
if (c == count):
    print("Embedding Overwriting Complete!")
# ### Downstream Task
node_embeddings_df = pd.DataFrame(node_embeddings)
userGraph_partitioned = StellarGraph.from_networkx(ig.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=node_embeddings_df)
start = time.time()
edge_splitter_test = EdgeSplitter(userGraph_partitioned)
graphsage_learning(edge_splitter_test)
print("time :", time.time() - start)
