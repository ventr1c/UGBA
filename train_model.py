#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI
#%%

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Pubmed', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--test_model', type=str, default='guard',
                    choices=['GCN','GAT','GraphSage','guard'],
                    help='Model used to attack')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')
parser.add_argument("--layer", type=int, default=2)

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
#%%
import os
from models.construct import model_construct

benign_model = model_construct(args,args.test_model,data,device).to(device) 
t_total = time.time()
print("Length of training set: {}".format(len(idx_train)))
benign_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
print("Training benign model Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

benign_ca = benign_model.test(data.x, data.edge_index, None, data.y,idx_clean_test)
print("Benign CA: {:.4f}".format(benign_ca))
benign_model = benign_model.cpu()
# %%