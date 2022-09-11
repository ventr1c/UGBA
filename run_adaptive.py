#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork,Reddit
from torch_geometric.utils import to_dense_adj,dense_to_sparse
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,select_target_nodes
import help_funcs
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='cora', help='Dataset',
                    choices=['cora','citeseer','pubmed'])
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
# parser.add_argument('--trigger_size', type=int, default=3,
#                     help='tirgger_size')
# parser.add_argument('--vs_ratio', type=float, default=0.01)
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=200, help='Number of epochs to train trigger generator.')

parser.add_argument('--load_benign_model', action='store_true', default=True,
                    help='Loading benign model if exists.')
# attack setting
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_ratio', type=float, default=0.01,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--target_test_nodes_num', type=float, default=200,
                    help="the number of of test nodes attached with 1 (independent) trigger, which is corretly classified and not belong to the target class")
parser.add_argument('--clean_test_nodes_num', type=float, default=200,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.1,
                    help="Threshold of prunning edges")
parser.add_argument('--homo_loss_weight', type=float, default=0,
                    help="Threshold of prunning edges")
parser.add_argument('--homo_boost_thrd', type=float, default=0.6,
                    help="Threshold of increase similarity")
parser.add_argument('--test_model', type=str, default='GraphSage',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=2,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

# args = parser.parse_known_args()[0]
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:1" if args.cuda else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


# In[2]:


#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

# if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
dataset = Planetoid(root='./data/', split="random", num_train_per_class=80, num_val=400, num_test=1000,                     name=args.dataset,transform=None)
# dataset = Reddit(root='./data/', transform=transform, pre_transform=None)
# dataset = classFlickr(root='./data/', transform=transform, pre_transform=None)

data = dataset[0].to(device)


# In[3]:


#%%
from torch_geometric.utils import to_undirected
# get the overall edge index of the graph
data.edge_index = to_undirected(data.edge_index)


# In[5]:


#%%  mask the test nodes
from utils import subgraph
# get the edge index used for training (except from test nodes) and 
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)

mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
idx_train =data.train_mask.nonzero().flatten()
idx_val = data.val_mask.nonzero().flatten()
idx_test = data.test_mask.nonzero().flatten()
# val_mask = node_idx[data.val_mask]
# labels = data.y[torch.bitwise_not(data.test_mask)]
# features = data.x[torch.bitwise_not(data.test_mask)]


# In[6]:


from models.GCN import GCN
from models.GAT import GAT
from models.GIN import GIN
from models.SAGE import GraphSage
def model_construct(args,model_name,data):
    if (model_name == 'GCN'):
        model = GCN(nfeat=data.x.shape[1],                    
                    nhid=args.hidden,                    
                    nclass= int(data.y.max()+1),                    
                    dropout=args.dropout,                    
                    lr=args.lr,                    
                    weight_decay=args.weight_decay,                    
                    device=device)
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(data.y.max()+1), 
                    heads=8,
                    dropout=args.dropout, 
                    lr=args.lr, 
                    weight_decay=args.weight_decay, 
                    device=device)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],                    
                        nhid=args.hidden,                    
                        nclass= int(data.y.max()+1),                    
                        dropout=args.dropout,                    
                        lr=args.lr,                    
                        weight_decay=args.weight_decay,                    
                        device=device)
    return model


# In[7]:


'''
train benign model
'''
import os
benign_modelpath = './modelpath/{}_{}_benign.pth'.format(args.model, args.dataset)
if(os.path.exists(benign_modelpath) and args.load_benign_model):
    # load existing benign model
    benign_model = torch.load(benign_modelpath)
    benign_model = benign_model.to(device)
    edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Loading benign {} model Finished!".format(args.model))
else:
    # benign_model = GCN(nfeat=data.x.shape[1],\
    #             nhid=args.hidden,\
    #             nclass= int(data.y.max()+1),\
    #             dropout=args.dropout,\
    #             lr=args.lr,\
    #             weight_decay=args.weight_decay,\
    #             device=device).to(device)
    benign_model = model_construct(args,args.model,data).to(device) 
    t_total = time.time()
    edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    benign_model.fit(data.x, data.edge_index, edge_weights, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training benign model Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Save trained model
    torch.save(benign_model, benign_modelpath)
    print("Benign model saved at {}".format(benign_modelpath))


# In[8]:


benign_output = benign_model(data.x, data.edge_index, edge_weights)
# benign4poison_output = benign_gcn(induct_x,induct_edge_index,induct_edge_weights)
# benign_ca = (benign_output.argmax(dim=1)[idx_test]==data.y[idx_test]).float().mean()
benign_ca = benign_model.test(data.x, data.edge_index, edge_weights, data.y,idx_test)
# benign4poison_ca = (benign4poison_output.argmax(dim=1)[idx_test]==atk_labels[idx_test]).float().mean()
print("Benign CA: {:.4f}".format(benign_ca))
atk_test_nodes, clean_test_nodes,poi_train_nodes = select_target_nodes(args,args.seed,benign_model,data.x, data.edge_index, edge_weights,data.y,idx_val,idx_test)
clean_test_ca = benign_model.test(data.x, data.edge_index, edge_weights, data.y,clean_test_nodes)
print("Benign CA on clean test nodes: {:.4f}".format(clean_test_ca))
# print("Benign for poisoning CA: {:.4f}".format(benign4poison_ca))
# print((benign_output.argmax(dim=1)[yx_nids]==args.target_class).float().mean())


# In[9]:


#%%
from models.backdoor import obtain_attach_nodes,Backdoor
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
# poison nodes' size
size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print(len(data.test_mask))
# here is randomly select poison nodes from unlabeled nodes
idx_attach = obtain_attach_nodes(unlabeled_idx,size)


# In[10]:


#%%
model = Backdoor(args,device)
print(args.epochs)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach)


# In[11]:


# %%
poison_x = model.poison_x.data
poison_edge_index = model.poison_edge_index.data
poison_edge_weights = model.poison_edge_weights.data
poison_labels = model.labels


# In[12]:


if(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'isolate'):
    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)


# In[13]:


print(len(torch.cat([idx_train,idx_attach])))
print(len(bkd_tn_nodes))
print(len(model.poison_edge_index.data[0]),len(poison_edge_index[0]))
# print(idx_attach & bkd_tn_nodes)
print(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))


# In[16]:


#%%
test_model = model_construct(args,args.test_model,data).to(device) 
if(args.test_model == 'GraphSage'):
    poison_adj = to_dense_adj(poison_edge_index, edge_attr=poison_edge_weights)
    poison_edge_index, poison_edge_weights = dense_to_sparse(poison_adj)
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=200,verbose=True)


# In[17]:


# gcn.eval()
# model.eval()
output = test_model(poison_x,poison_edge_index,poison_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
#%%
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

# idx_test = data.test_mask.nonzero().flatten()[:200]
# idx_test = list(set(data.test_mask.nonzero().flatten().tolist()) - set(atk_test_nodes))
# idx_atk = data.test_mask.nonzero().flatten()[200:].tolist()
# yt_nids = [nid for nid in idx_atk if data.y.tolist()==args.target_class] 
# yx_nids = torch.LongTensor(list(set(idx_atk) - set(yt_nids))).to(device)
atk_labels = poison_labels.clone()
atk_labels[atk_test_nodes] = args.target_class
clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,clean_test_nodes)
print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
#%% inject trigger on attack test nodes (idx_atk)
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(atk_test_nodes,poison_x,induct_edge_index,induct_edge_weights)
if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
# attack evaluation:
asr = test_model.test(induct_x,induct_edge_index,induct_edge_weights,atk_labels,atk_test_nodes)
ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,clean_test_nodes)
print("ASR: {:.4f}".format(asr))
print("CA: {:.4f}".format(ca))
# output = test_model(induct_x,induct_edge_index,induct_edge_weights)
# train_attach_rate = (output.argmax(dim=1)[atk_test_nodes]==args.target_class).float().mean()
# print("ASR: {:.4f}".format(train_attach_rate))

