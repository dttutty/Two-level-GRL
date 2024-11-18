from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/home/sqp17/Projects/Two-level-GRL/datasets/Citeseer', name='Citeseer')
data = dataset[0]
