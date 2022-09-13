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
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset',
                    choices=['Cora','Citeseer','Pubmed'])
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.0)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=200, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_ratio', type=float, default=0.01,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.3,
                    help="Threshold of prunning edges")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--attack_method', type=str, default='GTA',
                    choices=['Rand_Gene','Rand_Samp','GTA'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--trigger_prob', type=float, default=0.5,
                    help="The probability to generate the trigger's edges in random method")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')

# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))


#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

np.random.seed(11) # fix the random seed is important
dataset = Planetoid(root='./data/', \
                    name=args.dataset,\
                    transform=transform)

data = dataset[0].to(device)
# we build our own train test split 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(data,device)

#%%
from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]

# In[3]:
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)

# In[6]: 
import os
from models.backdoor import model_construct
benign_modelpath = './modelpath/{}_{}_benign.pth'.format(args.model, args.dataset)
if(os.path.exists(benign_modelpath) and args.load_benign_model):
    # load existing benign model
    benign_model = torch.load(benign_modelpath)
    benign_model = benign_model.to(device)
    edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Loading benign {} model Finished!".format(args.model))
else:
    benign_model = model_construct(args,args.model,data,device).to(device) 
    t_total = time.time()
    print("Length of training set: {}".format(len(idx_train)))
    benign_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
    print("Training benign model Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Save trained model
    # torch.save(benign_model, benign_modelpath)
    # print("Benign model saved at {}".format(benign_modelpath))

# In[7]:

benign_ca = benign_model.test(data.x, data.edge_index, None, data.y,idx_clean_test)
print("Benign CA: {:.4f}".format(benign_ca))
# In[9]:

from sklearn_extra import cluster
from models.backdoor import obtain_attach_nodes,Backdoor,obtain_attach_nodes_by_influential,obtain_attach_nodes_by_cluster
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)

# here is randomly select poison nodes from unlabeled nodes
if(args.selection_method == 'none'):
    idx_attach = obtain_attach_nodes(unlabeled_idx,size)
elif(args.selection_method == 'loss' or args.selection_method == 'conf'):
    idx_attach = obtain_attach_nodes_by_influential(args,benign_model,unlabeled_idx.cpu().tolist(),data.x,data.edge_index,edge_weights,data.y,device,size,selected_way=args.selection_method)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster'):
    # construct GCN encoder
    encoder_modelpath = './modelpath/{}_{}_benign.pth'.format('GCN_Encoder', args.dataset)
    if(os.path.exists(encoder_modelpath) and args.load_benign_model):
        # load existing benign model
        gcn_encoder = torch.load(encoder_modelpath)
        gcn_encoder = gcn_encoder.to(device)
        edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Loading {} encoder Finished!".format(args.model))
    else:
        gcn_encoder = model_construct(args,'GCN_Encoder',data,device).to(device) 
        t_total = time.time()
        # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Length of training set: {}".format(len(idx_train)))
        gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
        print("Training encoder Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # # Save trained model
        # torch.save(gcn_encoder, encoder_modelpath)
        # print("Encoder saved at {}".format(encoder_modelpath))
    # test gcn encoder 
    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()
    kmeans = cluster.KMedoids(n_clusters=nclass,method='pam')
    kmeans.fit(encoder_x.detach().cpu().numpy())
    idx_attach = obtain_attach_nodes_by_cluster(args,kmeans,unlabeled_idx.cpu().tolist(),encoder_x,data.y,device,size)
    idx_attach = torch.LongTensor(idx_attach).to(device)

# In[10]:
# train trigger generator 
model = Backdoor(args,device)
if(args.attack_method == 'GTA'):
    model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()
elif(args.attack_method == 'Rand_Gene' or args.attack_method == 'Rand_Samp'):
    model.fit_rand(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

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

print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
#%%
test_model = model_construct(args,args.test_model,data,device).to(device) 
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=200,verbose=False)

output = test_model(poison_x,poison_edge_index,poison_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
#%%
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
print("accuracy on clean test nodes: {:.4f}".format(clean_acc))

# %% inject trigger on attack test nodes (idx_atk)'''
induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights)
# do pruning in test datas'''
if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
# attack evaluation

output = test_model(induct_x,induct_edge_index,induct_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
print("ASR: {:.4f}".format(train_attach_rate))
ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
print("CA: {:.4f}".format(ca))

# %%
