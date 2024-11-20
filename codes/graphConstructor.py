import pandas as pd
import networkx as nx
import igraph
import stellargraph as sg
from pathlib import Path
from scipy import io
from stellargraph import datasets, StellarGraph

DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


def graphloader(dataset_str):
    if dataset_str == 'Foursquare':
        file = DATASET_DIR / "dataset_WWW_friendship_new.txt"
        node_features = pd.read_csv(DATASET_DIR / "node_features_encoded.csv", index_col=0)
        country_degree = pd.concat([node_features['countrycode_encoded'], node_features['degree']],axis=1)
        g = nx.read_edgelist(file , nodetype = int, edgetype='Freindship')
        ig = igraph.Graph.from_networkx(g) # NetworkX to igraph 
        G = StellarGraph.from_networkx(ig.to_networkx(), node_type_default = "user", edge_type_default = "friendship", node_features = country_degree)
        node_subjects = None # Foursquare dataset has no labels
        
    elif dataset_str == 'Flickr':
        file = io.loadmat(DATASET_DIR / "Flickr.mat")
        G = nx.from_scipy_sparse_matrix(file['Network'])
        fea_mat = file['Attributes'].todense()
        fea_mat = pd.DataFrame(fea_mat)
        node_subjects = []
        for i in file['Label']:
            node_subjects.append(i[0])
        node_subjects = pd.Series(node_subjects)
        G = StellarGraph.from_networkx(G, node_features = fea_mat)

    elif dataset_str == 'PubMed':
        dataset = datasets.PubMedDiabetes()
        G, node_subjects = dataset.load()

    elif dataset_str == 'CiteSeer':
        dataset = datasets.CiteSeer()
        G, node_subjects = dataset.load()

    # Print Graph Information 
    try:
        if G:
            print("####### %s Graph Construction Complete #######" %(dataset_str))
            print()
            print(G.info())
            return G, node_subjects
    # Error         
    except NameError:
        print("There are no dataset ' %s '" %(dataset_str))
