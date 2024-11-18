#!/usr/bin/env python
# coding: utf-8
# 3) community + node: Major community에 대해 Local GRL 수행 + Reduced Graph에 대해 Global GRL 수행 후 임베딩 통합
#
# => 정확도 1)에 근접, 병렬처리 가능(처리속도 상)
# * Parallel Local GRL for Major + Global Reduced GRL for Minor 수행 시간 측정
# 基础数据处理库
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict
import multiprocessing as mp
from multiprocessing import Pool

# 图处理库
import networkx as nx
from igraph import Graph
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter, BiasedRandomWalk

# 嵌入与深度学习库
from gensim.models import Word2Vec
from tensorflow import keras

# 机器学习和评估库
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

mp.cpu_count()
# Edge between two user node (607,333 friendships)
file = '/home/sqp17/Projects/Two-level-GRL/datasets/dataset_WWW_friendship_new.txt'
g = nx.read_edgelist(file, nodetype=int, edgetype='Freindship')
ig = Graph.from_networkx(g)  # NetworkX to igraph
ig.vs["id"] = ig.vs["_nx_name"]
ig.es["weight"] = [1.0]*ig.ecount()
node_features_encoded = pd.read_csv(
    "/home/sqp17/Projects/Two-level-GRL/datasets/node_features_encoded.csv", index_col=0)
country_degree = pd.concat(
    [node_features_encoded['countrycode_encoded'], node_features_encoded['degree']], axis=1)
userGraph_country_deg = StellarGraph.from_networkx(ig.to_networkx(
), node_type_default="user", edge_type_default="friendship", node_features=country_degree)
print(userGraph_country_deg.info())  # 전체 그래프 load
# ### 1. Community 단위로 RL 적용 (local RL) - 병렬 처리 가능
LP = Graph.community_label_propagation(ig)  # 커뮤니티 디텍션
LP.summary()
minor_nodes_num = 0
major_comm_num = 0
for i in LP:
    if len(i) < 100:
        minor_nodes_num += len(i)
    else:
        major_comm_num += 1
print("100보다 작은 커뮤니티의 노드 수: ", minor_nodes_num)
print("100보다 큰 커뮤니티 개수: ", major_comm_num)
print("Reduced graph 노드 수: ", major_comm_num+minor_nodes_num)
minor_nodes_num = 0
major_comm_num = 0
for i in LP:
    if len(i) < 100:
        minor_nodes_num += len(i)
minor_nodes_num
major_nodes_num = 0
for i in LP:
    if len(i) >= 100:
        major_nodes_num += len(i)
major_nodes_num
# Node2Vec Hyper-parameter Settings
p = 1.0  # p가 낮을 수록 좁은 지역을 보고 q가 낮을수록 넓은 지역을 봅니다.
q = 1.0
dimensions = 256
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1
workers = mp.cpu_count()


def sub_Node2Vec(subgraphList):
    if len(subgraphList) < 100:
        return None

    subgraph = ig.induced_subgraph(
        subgraphList, implementation="create_from_scratch")
    isin_filter = node_features_encoded['userID'].isin(subgraph.vs['id'])

    subgraph_features = node_features_encoded[isin_filter]
    subgraph_country_degree = pd.concat(
        [subgraph_features['countrycode_encoded'], subgraph_features['degree']], axis=1)
    subgraph_country_degree.reset_index(drop=True, inplace=True)

    subgraph_ = StellarGraph.from_networkx(subgraph.to_networkx(
    ), node_type_default="user", edge_type_default="friendship", node_features=subgraph_country_degree)
  #  print("Node개수: ",len(subgraph_.nodes()))

    #########################################################################
    rw = BiasedRandomWalk(subgraph_)
    walks = rw.run(subgraph_.nodes(), n=num_walks,
                   length=walk_length, p=p, q=q)
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=num_iter,
    )
    return model.wv


node_embeddings = defaultdict(lambda: np.zeros(dimensions))  # Initialization
# In[ ]:
num_cores = mp.cpu_count()
pool_obj = mp.Pool(1)
start = time.time()
sub_node_embeddings_list = pool_obj.map(sub_Node2Vec, list(LP))
localGRL_time = start - time.time()
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")
pool_obj = mp.Pool(5)
start = time.time()
sub_node_embeddings_list = pool_obj.map(sub_Node2Vec, list(LP))
localGRL_time = start - time.time()
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")
pool_obj = mp.Pool(10)
start = time.time()
sub_node_embeddings_list = pool_obj.map(sub_Node2Vec, list(LP))
localGRL_time = start - time.time()
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")
pool_obj = mp.Pool(15)
start = time.time()
sub_node_embeddings_list = pool_obj.map(sub_Node2Vec, list(LP))
localGRL_time = start - time.time()
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")
pool_obj = mp.Pool(20)
start = time.time()
sub_node_embeddings_list = pool_obj.map(sub_Node2Vec, list(LP))
localGRL_time = start - time.time()
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")
pool_obj = mp.Pool(num_cores)
start = time.time()
sub_node_embeddings_list = pool_obj.map(sub_Node2Vec, list(LP))
localGRL_time = start - time.time()
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")

for community_idx in range(len(LP)):
    if sub_node_embeddings_list[community_idx] is not None:
        sub_node_embeddings = sub_node_embeddings_list[community_idx]
        j = 0
        for i in LP[community_idx]:
            node_embeddings[i] = sub_node_embeddings[j]
            j += 1
len(node_embeddings)
# In[ ]:
''' 
@@@ NO PARALLEL PROCESSING VERSION @@@
start = time.time()
for community_idx in range(len(LP)):
    # Intra Community Embedding 
    if len(LP[community_idx]) >= 100:
        sub_node_embeddings = sub_Node2Vec(LP[community_idx],"Sub-graph no.["+str(community_idx)+"]")
        print(len(sub_node_embeddings))
        
        # 전체 그래프에 대한 GraphSAGE에 의해 도출된 feature를 아예 덮어쓰는 것. 
        j=0
        for i in LP[community_idx]:
          #  print(i, end=' ')
            node_embeddings[i] = sub_node_embeddings[j]
            j += 1
localGRL_time = time.time() - start
print("@@@ Major Community Local GRL time :", localGRL_time, " @@@")
'''
# ### Community는 하나의 node로 변환함으로써 축소된 Global graph 생성
membership = LP.membership
counter = Counter(membership).most_common()
minor_vertexID = []
# community size 가 100보다 작으면 minor vertex ID list에 추가
for c in counter:
    if c[1] < 100:
        minor_vertexID.append(c[0])
print(len(minor_vertexID))
# Minor community 인 애들은 노드 살려놔야하니까 membership 다시 부여하기
new_id = len(LP)
for i in range(len(membership)):
    if membership[i] in minor_vertexID:
        membership[i] = new_id
        new_id += 1
idx_map = {}
n = 0
for i in sorted(dict(Counter(membership))):
    idx_map[i] = n
    n += 1
# 인덱스 순서대로 초기화
new_idx = 0
for i in range(len(membership)):
    membership[i] = idx_map[membership[i]]
ig.contract_vertices(membership, combine_attrs="mean")
# ### RL 적용 (global RL) - 기존 graph에 비해 연산 크게 축소
reduced_globalG = StellarGraph.from_networkx(
    ig.to_networkx(), node_type_default="user", edge_type_default="friendship")
print(reduced_globalG.info())  # 전체 그래프 load


def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    print(f"Number of random walks for '{name}': {len(walks)}")
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=num_iter,
    )
    return model.wv


start = time.time()
reduced_globalG_embedding = node2vec_embedding(
    reduced_globalG, "Reduced Global Graph")
globalGRL_time = time.time() - start
print("@@@ Global reduced GRL time :", globalGRL_time, " @@@")
# global graph에 대한 RL 적용 결과 붙이기
start = time.time()
for community_idx in range(len(LP)):
    # Intra Community Embedding
    if len(LP[community_idx]) < 100:
        j = 0
        for i in LP[community_idx]:
            node_embeddings[i] = reduced_globalG_embedding[j]
            j += 1
print("time :", time.time() - start)
cnt = 0
for nodes in node_embeddings.values():
    if nodes[0] == 0.0:  # 임베딩 업데이트 안된 노드 있는지 확인
        cnt += 1
cnt
# ### 전체 그래프에 대해서 Link Prediction 결과 확인
# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(userGraph_country_deg)
# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global"
)
print(graph_test.info())
# Do the same process to compute a training subset from within the test graph
edge_splitter_train = EdgeSplitter(graph_test, userGraph_country_deg)
graph_train, examples, labels = edge_splitter_train.train_test_split(
    p=0.1, method="global"
)
(
    examples_train,
    examples_model_selection,
    labels_train,
    labels_model_selection,
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)
print(graph_train.info())
pd.DataFrame(
    [
        (
            "Training Set",
            len(examples_train),
            "Train Graph",
            "Test Graph",
            "Train the Link Classifier",
        ),
        (
            "Model Selection",
            len(examples_model_selection),
            "Train Graph",
            "Test Graph",
            "Select the best Link Classifier model",
        ),
        (
            "Test set",
            len(examples_test),
            "Test Graph",
            "Full Graph",
            "Evaluate the best Link Classifier",
        ),
    ],
    columns=("Split", "Number of Examples",
             "Hidden from", "Picked from", "Use"),
).set_index("Split")
# 1. link embeddings


def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node[src], transform_node[dst])
        for src, dst in link_examples
    ]
# 2. training classifier


def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(
        Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])
# 3. and 4. evaluate classifier


def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


def run_link_prediction(binary_operator, embedding_train):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )
    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
results = [run_link_prediction(op, node_embeddings) for op in binary_operators]
best_result = max(results, key=lambda result: result["score"])
print(f"Best result from '{best_result['binary_operator'].__name__}'")
pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"])
     for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")
for result in results:
    test_score = evaluate_link_prediction_model(
        result["classifier"],
        examples_test,
        labels_test,
        node_embeddings,
        result["binary_operator"],
    )
    print(
        f"ROC AUC score on test set using '{result['binary_operator'].__name__}': {test_score}"
    )
print("Total Time: ", localGRL_time + globalGRL_time)
print("Total Time: ", -1*localGRL_time + globalGRL_time)
