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
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Pubmed', help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--budget', type=float, default=0.001)
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')

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
dataset = Planetoid(root='./data/', split="random", num_train_per_class=80, num_val=400, num_test=400, \
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
# val_mask = node_idx[data.val_mask]
# labels = data.y[torch.bitwise_not(data.test_mask)]
# features = data.x[torch.bitwise_not(data.test_mask)]
#%%
from models.backdoor import obtain_attach_nodes,Backdoor

unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
size = int((len(data.test_mask)-data.test_mask.sum())*args.budget)
idx_attach = obtain_attach_nodes(unlabeled_idx,size)

#%%

model = Backdoor(args,device)
print(args.epochs)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach)
#%% 


# %%
poison_x = model.poison_x.data
poison_edge_index = model.poison_edge_index.data
poison_edge_weights = model.poison_edge_weights.data
poison_labels = model.labels

#%%
from models.GCN import GCN

# poison_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
# poison_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])


gcn = GCN(nfeat=data.x.shape[1],\
            nhid=args.hidden,\
            nclass= int(data.y.max()+1),\
            dropout=args.dropout,\
            lr=args.lr,\
            weight_decay=args.weight_decay,\
            device=device).to(device)

#%%
gcn.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, torch.cat([idx_train,idx_attach]), idx_val,train_iters=args.epochs,verbose=True)

# %%
output = gcn(poison_x,poison_edge_index,poison_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
#%%
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

idx_test = data.test_mask.nonzero().flatten()[:200]
idx_atk = data.test_mask.nonzero().flatten()[200:]
clean_acc = gcn.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_test)
print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
#%% inject trigger on attack test nodes (idx_atk)
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights)
output = gcn(induct_x,induct_edge_index,induct_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
print("ASR: {:.4f}".format(train_attach_rate))
# %%
