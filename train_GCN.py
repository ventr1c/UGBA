#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
#%%

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Cora', help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
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

# if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
dataset = Planetoid(root='./data/', split="random", num_train_per_class=40, num_val=400, num_test=400, \
                    name=args.dataset,transform=transform)
data = dataset[0].to(device)
#%%
from torch_geometric.utils import to_undirected
data.edge_index = to_undirected(data.edge_index)
#%%  mask the test nodes
from utils import subgraph
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
idx_train =data.train_mask.nonzero().flatten()
idx_val = data.val_mask.nonzero().flatten()
#%%
model = GCN(nfeat=data.x.shape[1],\
            nhid=args.hidden,\
            nclass= int(data.y.max()+1),\
            dropout=args.dropout,\
            lr=args.lr,\
            weight_decay=args.weight_decay,\
            device=device,layer=args.layer).to(device)

#%%
model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs)
result = model.test(data.x, data.edge_index, data.edge_attr,data.y, data.test_mask)

print("test result: {}".format(result))
# %%