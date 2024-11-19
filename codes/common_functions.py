# common_functions.py
from tensorflow import keras
import stellargraph as sg
import pickle
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.data import EdgeSplitter, BiasedRandomWalk
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter, UnsupervisedSampler
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing as mp
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import numpy as np
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import multiprocessing as mp
import pandas as pd
from sklearn.pipeline import Pipeline
from igraph import Graph
import networkx as nx
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import GraphSAGE
from tensorflow import keras
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV


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


def load_graph_data(EDGE_LIST_FILE, NODE_FEATURES_FILE):
    """加载图数据与节点特征"""
    g = nx.read_edgelist(EDGE_LIST_FILE, nodetype=int, edgetype='Freindship')
    ig = Graph.from_networkx(g)
    ig.vs["id"] = ig.vs["_nx_name"]
    ig.es["weight"] = [1.0] * ig.ecount()
    
    node_features_encoded = pd.read_csv(NODE_FEATURES_FILE, index_col=0)
    country_degree = node_features_encoded[['countrycode_encoded', 'degree']]
    
    user_graph = StellarGraph.from_networkx(
        ig.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=country_degree
    )
    return ig, user_graph, node_features_encoded


def sub_Node2Vec(
    ig, 
    node_features_encoded, 
    subgraphList, 
    ignore_minor_graph=False, 
    num_walks=10, 
    walk_length=80, 
    p=1.0, 
    q=1.0, 
    dimensions=256, 
    window_size=10, 
    workers=None, 
    num_iter=1
):
    """
    计算子图的节点嵌入。

    Parameters:
    - ig: 原始图 (igraph 对象)。
    - node_features_encoded: 包含节点特征的 DataFrame。
    - subgraphList: 子图节点列表。
    - ignore_minor_graph: 是否忽略较小的子图。
    - num_walks: 每个节点的随机游走次数。
    - walk_length: 随机游走的步长。
    - p: 随机游走的反向控制参数。
    - q: 随机游走的远距离控制参数。
    - dimensions: 嵌入的维度。
    - window_size: Word2Vec 窗口大小。
    - workers: 使用的 CPU 核心数。
    - num_iter: Word2Vec 的训练迭代次数。

    Returns:
    - model.wv: Word2Vec 模型的嵌入向量。
    """
    # 设置默认 workers
    if workers is None:
        workers = mp.cpu_count()

    # 忽略小图逻辑
    if ignore_minor_graph and len(subgraphList) < 100:
        return None

    # 提取子图
    subgraph = ig.induced_subgraph(subgraphList, implementation="create_from_scratch")

    # 筛选子图节点的特征
    isin_filter = node_features_encoded['userID'].isin(subgraph.vs['id'])
    subgraph_features = node_features_encoded[isin_filter]
    subgraph_country_degree = subgraph_features[['countrycode_encoded', 'degree']].reset_index(drop=True)

    # 将子图转换为 StellarGraph
    subgraph_ = StellarGraph.from_networkx(
        subgraph.to_networkx(),
        node_type_default="user",
        edge_type_default="friendship",
        node_features=subgraph_country_degree
    )

    # 打印子图信息
    print(subgraph_.info())

    # 随机游走
    rw = BiasedRandomWalk(subgraph_)
    walks = rw.run(
        nodes=subgraph_.nodes(), 
        n=num_walks, 
        length=walk_length, 
        p=p, 
        q=q
    )
    print(f"Number of random walks: {len(walks)}")

    # 使用 Word2Vec 训练嵌入
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,  # Skip-gram 模型
        workers=workers,
        epochs=num_iter
    )

    # 返回嵌入向量
    return model.wv


def subGraphSAGE(
    ig, 
    subgraphList, 
    node_features_encoded, 
    verbose=1, 
    dropout=0.0, 
    batch_size=20, 
    num_samples=(20, 10), 
    layer_sizes=(50, 50), 
    walk_length=5, 
    num_walks=1, 
    normalize="l2"
):
    """
    Perform GraphSAGE embedding on a subgraph.

    Parameters:
    - ig: The input graph (igraph object).
    - subgraphList: List of nodes to extract the subgraph.
    - node_features_encoded: Pandas DataFrame containing node features.
    - verbose: Verbosity level for Keras model prediction.
    - dropout: Dropout rate for the GraphSAGE model.
    - batch_size: Batch size for data generators.
    - num_samples: Number of neighbors to sample at each layer.
    - layer_sizes: Number of units in each GraphSAGE layer.
    - walk_length: Length of random walks for unsupervised sampling.
    - num_walks: Number of random walks per node.
    - normalize: Normalization strategy for GraphSAGE.

    Returns:
    - sub_node_embeddings: Numpy array containing the node embeddings.
    """

    # Step 1: Create the induced subgraph
    subgraph = ig.induced_subgraph(
        subgraphList, implementation="create_from_scratch"
    )

    # Step 2: Filter node features for the subgraph
    isin_filter = node_features_encoded['userID'].isin(subgraph.vs['id'])
    subgraph_features = node_features_encoded[isin_filter]
    subgraph_country_degree = subgraph_features[['countrycode_encoded', 'degree']].reset_index(drop=True)

    # Step 3: Convert to StellarGraph
    subgraph_ = StellarGraph.from_networkx(
        subgraph.to_networkx(), 
        node_type_default="user", 
        edge_type_default="friendship", 
        node_features=subgraph_country_degree
    )

    # Check if the subgraph is large enough for training
    num_nodes = len(subgraph_.nodes())
    if num_nodes > 100:
        print(f"Number of nodes: {num_nodes}")

    # Step 4: Prepare data generators
    subnodes = list(subgraph_.nodes())
    sub_unsupervised_samples = UnsupervisedSampler(
        subgraph_, nodes=subnodes, length=walk_length, number_of_walks=num_walks
    )
    sub_generator = GraphSAGELinkGenerator(subgraph_, batch_size, num_samples)
    sub_train_gen = sub_generator.flow(sub_unsupervised_samples)

    # Step 5: Define the GraphSAGE model
    sub_graphsage = GraphSAGE(
        layer_sizes=layer_sizes, 
        generator=sub_generator, 
        bias=True, 
        dropout=dropout, 
        normalize=normalize
    )

    # Step 6: Build the embedding model
    x_inp, x_out = sub_graphsage.in_out_tensors()
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    sub_embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # Step 7: Generate embeddings for all nodes in the subgraph
    sub_node_ids = subgraph_.nodes()
    sub_node_gen = GraphSAGENodeGenerator(
        subgraph_, batch_size, num_samples
    ).flow(sub_node_ids)

    sub_node_embeddings = sub_embedding_model.predict(
        sub_node_gen, workers=4, verbose=verbose
    )

    return sub_node_embeddings




def graphsage_learning(
    edge_splitter_test,
    graph=None,
    epochs=50,
    need_dump=False,
    batch_size=20,
    num_samples=(20, 10),
    layer_sizes=(50, 50),
    learning_rate=1e-3,
    dropout=0.3
    ):
    # Helper function to evaluate and print metrics
    def evaluate_and_print_metrics(model, data_flow, description):
        metrics = model.evaluate(data_flow, verbose=0)
        print(f"\n{description} Metrics:")
        for name, val in zip(model.metrics_names, metrics):
            print(f"\t{name}: {val:.4f}")
        return metrics

    # Step 1: Train-test split for edge data
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global", keep_connected=True
    )
    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.1, method="global", keep_connected=True
    )

    # Step 2: Data generators for training and testing
    train_gen = GraphSAGELinkGenerator(
        G_train, batch_size, num_samples, weighted=True
    )
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
    test_gen = GraphSAGELinkGenerator(
        G_test, batch_size, num_samples, weighted=True
    )
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

    # Step 3: Define the GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=dropout
    )
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.binary_crossentropy,
        metrics=["acc"],
    )

    # Step 4: Evaluate the initial model
    evaluate_and_print_metrics(model, train_flow, "Train Set (Initial)")
    evaluate_and_print_metrics(model, test_flow, "Test Set (Initial)")

    # Step 5: Train the model
    history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

    # Plot training history
    sg.utils.plot_history(history)

    # Optional: Save training history
    if need_dump:
        with open("trainHistoryDict.pkl", "wb") as file_pi:
            pickle.dump(history.history, file_pi)

    # Step 6: Evaluate the trained model
    evaluate_and_print_metrics(model, train_flow, "Train Set (Trained)")
    evaluate_and_print_metrics(model, test_flow, "Test Set (Trained)")

    # Step 7: Generate node embeddings if graph is provided
    if graph is not None:
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

        node_ids = graph.nodes()
        node_gen = GraphSAGENodeGenerator(
            graph, batch_size, num_samples
        ).flow(node_ids)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

        return node_embeddings

    return None



def node2vec_embedding(graph, name, dimensions=256, num_walks=10, walk_length=80, p=1.0, q=1.0, window_size=10, num_iter=1, workers=None):
    if workers is None:
        workers = mp.cpu_count()
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
#    def get_embedding(u):
#        return model.wv[u]
#    return get_embedding
    return model.wv


def run_link_prediction(binary_operator, embedding_train,  examples_train, labels_train, examples_model_selection, labels_model_selection):
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
    
    
def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0

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



