
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import imp
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
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
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('-- ', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--use_vs_number', action='store_true', default=False,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=0,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
args = parser.parse_args()
# args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
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


# In[9]:

from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.construct import model_construct
import heuristic_selection as hs

# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()

models = ['GCN','GAT', 'GraphSage']
# total_overall_asr = 0
total_overall_ca = 0
for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    # seeds = [args.seed]
    overall_ca = 0
    for seed in seeds:
        args.seed = seed

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print(args)
        #%%
        test_model = model_construct(args,args.test_model,data,device).to(device) 
        # TODO: add multiple time seeds
        if(args.defense_mode == 'prune'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights = prune_unrelated_edge(args,train_edge_index,train_edge_weights,data.x,device,large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
        elif(args.defense_mode == 'isolate'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights, rel_nodes = prune_unrelated_edge_isolated(args,train_edge_index,train_edge_weights,data.x,device,large_graph=False)
            idx_tn_nodes = torch.LongTensor(list(set(idx_train) - set(rel_nodes))).to(device)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        else:
            test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
        # output = test_model(data.x,data.edge_index,None)
        # train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        # print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        # torch.cuda.empty_cache()
        #%%
        # induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        # induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
            induct_edge_weights = torch.ones([data.edge_index.shape[1]]).to(device)
            induct_edge_index, induct_edge_weights = prune_unrelated_edge(args,data.edge_index,induct_edge_weights,data.x,device,large_graph=False)
            clean_acc = test_model.test(data.x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        else:
            clean_acc = test_model.test(data.x,data.edge_index,None,data.y,idx_clean_test)
        # test_model = test_model.cpu()

        print("accuracy on clean test nodes: {:.4f} ({} model, Seed: {})".format(clean_acc,args.test_model, args.seed))
        overall_ca += clean_acc

    overall_ca = overall_ca/len(seeds)
    print("Overall Clean Accuracy: {:.4f} ({} model)".format(overall_ca, args.test_model))

    # total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    test_model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
# total_overall_asr = total_overall_asr/len(models)
total_overall_ca = total_overall_ca/len(models)
# print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))