import random
import time
from collections import defaultdict, Counter
import argparse
import numpy as np
import pandas as pd
from graphConstructor import graphloader
from communityDetection import communityDetector
from igraph import *
import stellargraph as sg

#.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
random.seed(0)
np.random.seed(0)
def main():
    
    parser = argparse.ArgumentParser("Two-Level GRL")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Foursquare",
    )

    parser.add_argument("--size_thresh", type=int, default=100)
    parser.add_argument("--model", type=str, default="graphsage")
    parser.add_argument("--task", type=str, default="node")
    parser.add_argument("--cd_algo", type=str, default="LP")

    
    #args, unknown = argument.parse_args()
    args = parser.parse_args()
    
    
    # Graph Construction 
    G, node_subjects = graphloader(args.dataset)

    # Convert the data structure from StellarGraph -> NetworkX -> Igraph
    ig = Graph.from_networkx(G.to_networkx())

    # Community Detection
    cd_algo = communityDetector(ig, args.cd_algo)

    fea_mat = pd.DataFrame(G.node_features(), index= G.nodes())

    # Preserve the connections for original node information of super nodes
    original_edges = ig.get_edgelist()
    super_node_id = defaultdict(list)
    super_node_edges = defaultdict(set) # { super_node_id: [edge_list] }
    
    size_thresh = args.size_thresh

    for commu in range(len(cd_algo)):
        if len(cd_algo[commu]) >= size_thresh: # If major community
            for node in cd_algo[commu]: # for each node 
                super_node_id[commu].append(node) # the index of nodes which should be converted to super node
                for edges in original_edges: # check edges
                    if edges[0] == node: # If it includes current node
                        super_node_edges[commu].add(edges[1]) # add the neighborhood node index number
                    elif edges[1] == node: 
                        super_node_edges[commu].add(edges[0])
    
    # create a mapping dictionary (node id: nx_name) from igraph.
    node_dict = {}
    for idx in range(ig.vcount()):
        node_dict[idx] = ig.vs['_nx_name'][idx]

    # CaaN Graph Construction
    CaaN = ig.copy()
    membership = cd_algo.membership
    counter = Counter(membership).most_common()

    minor_commuID = []
    # If the community size is less than threshold, add to minor community ID list
    for c in counter:
        if c[1] < size_thresh:
            minor_commuID.append(c[0])

    # Re-assign membership for minor community nodes
    new_id = len(cd_algo)
    for i in range(len(membership)):
        if membership[i] in minor_commuID:
            membership[i] = new_id
            new_id += 1
            
    idx_map = {}
    n = 0
    for i in sorted(dict(Counter(membership))):
        idx_map[i] = n
        n+=1
        
    # Initialization by index
    new_idx = 0
    for i in range(len(membership)):
        membership[i] = idx_map[membership[i]]
    
    print(counter)
    CaaN.contract_vertices(membership, combine_attrs="first")
    CaaN.simplify(combine_edges="ignore") 
    print("Caan Graph Information : ")
    print(CaaN.summary())
    
    # Remove minor nodes from CaaN except for super nodes
    superG = CaaN.copy()
    cnt = 0
    deleted = []
    for v in range(superG.vcount()):
        if superG.vs['_nx_name'][v] in minor_commuID: #minor_nodes_features
            deleted.append(v)
            cnt += 1
    print(cnt)
    superG.delete_vertices(deleted)

    # Global GRL 
    CaaN_fea_mat = fea_mat[fea_mat.index.isin(CaaN.vs['_nx_name'])] # extract features of subgraphs
    CaaN_sg = sg.StellarGraph.from_networkx(CaaN.to_networkx(), node_features = CaaN_fea_mat)
    CaaN_subjects = node_subjects[node_subjects.index.isin(CaaN.vs['_nx_name'])].reset_index(drop=True) # Label of subgraphs
    
    
    def local_grl(model,X_base):
        start = time.time()
        node_embeddings = X_base
        # Local GRL 
        for commu in range(len(cd_algo)):
            if len(cd_algo[commu]) >= size_thresh: # If Major Community
                sub_node_embeddings = model.subgraph_learning(ig, cd_algo[commu], fea_mat, node_subjects)
                print("cd_algo's length : {}, sub_node_emb's shape : {}".format(len(cd_algo[commu]),len(sub_node_embeddings)))

                # Overwrite from subgraph embeddings
                j=0
                for i in cd_algo[commu]:
                    node_embeddings[i] = sub_node_embeddings[j]
                    j += 1
    

        print("Local GRL Time : ",time.time() - start)
        return node_embeddings
    
    

    # Base model and Global GRL Training 
    if args.model == 'gcn':
        from gcn import GCNModel
        if args.task == 'node': 
            X_base = GCNModel.node_classification(G, node_subjects) # Base model 
            X_CaaN = GCNModel.node_classification(CaaN_sg, CaaN_subjects) # Global GRL
        elif args.task == 'link':
            X_base = GCNModel.link_prediction(G, args)
            X_CaaN = GCNModel.link_prediction(CaaN_sg, args)
        local_grl(GCNModel ,X_base)

    elif args.model == 'graphsage':
        from graphsage import GraphSAGEModel
        if args.task == 'node': 
            X_base = GraphSAGEModel.node_classification(G, node_subjects)
            local_grl(GraphSAGEModel ,X_base)
            X_CaaN = GraphSAGEModel.node_classification(CaaN_sg, CaaN_subjects)
        elif args.task == 'link':
            X_base = GraphSAGEModel.link_prediction(G)
            X_CaaN = GraphSAGEModel.link_prediction(CaaN_sg)
            local_grl(GraphSAGEModel ,X_base)
        

    elif args.model == 'gat':
        from gat import GATModel
        if args.task == 'node': 
            X_base = GATModel.node_classification(G, node_subjects)
            X_CaaN = GATModel.node_classification(CaaN_sg, CaaN_subjects)
        elif args.task == 'link':
            X_base = GATModel.link_prediction(G)
            X_CaaN = GATModel.link_prediction(CaaN_sg)
        local_grl(GATModel ,X_base)

    elif args.model == 'node2vec':
        from node2vec import Node2VecModel
        if args.task == 'node': 
            X_base = Node2VecModel.node_classification(G, node_subjects)
            X_CaaN = Node2VecModel.node_classification(CaaN_sg, CaaN_subjects)
        elif args.task == 'link':
            X_base = Node2VecModel.link_prediction(G)
            X_CaaN = Node2VecModel.link_prediction(CaaN_sg)
        local_grl(Node2VecModel ,X_base)

    
    
    
if __name__ == "__main__":
    main()
