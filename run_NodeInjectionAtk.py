#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import time
import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI
from torch_geometric.utils import dense_to_sparse

from ogb.nodeproppred import PygNodePropPredDataset
# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
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
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=200, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_ratio', type=float, default=0.005,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="median",
                    choices=['prune', 'isolate', 'none', 'guard', 'median'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.15,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=3,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--attack_method', type=str, default='Basic',
                    choices=['Rand_Gene','Rand_Samp','Basic','None','TDGIA','AGIA'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--trigger_prob', type=float, default=0.5,
                    help="The probability to generate the trigger's edges in random method")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN','GNNGuard','MedianGCN','RobustGCN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

# np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(args.seed)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

np.random.seed(11) # fix the random seed is important
if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'PPI'):
    dataset = PPI(root='./data/PPI/', 
                split='train', transform=None)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 


data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
#%%
from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]

# In[3]:

# In[6]: 
import os
from models.construct import model_construct
from torch_sparse import SparseTensor
if(args.test_model == 'RobustGCN'):
    benign_model = model_construct(args,args.test_model,data,device).to(device) 
    train_edge_weights = torch.ones([train_edge_index.shape[1]],device=device,dtype=torch.float)
    sp_train_adj = SparseTensor.from_edge_index(train_edge_index,None,sparse_sizes = ([data.x.shape[0],data.x.shape[0]]))
    print("sp_train_adj",sp_train_adj)
    benign_model.fit(data.x, sp_train_adj, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
    sp_test_adj = SparseTensor.from_edge_index(data.edge_index,None,sparse_sizes = ([data.x.shape[0],data.x.shape[0]]))
    benign_ca = benign_model.test(data.x, sp_test_adj, None, data.y,idx_clean_test)
    print("Benign CA: {:.4f}".format(benign_ca))
else:
    benign_model = model_construct(args,args.test_model,data,device).to(device) 
    t_total = time.time()
    print("Length of training set: {}".format(len(idx_train)))
    benign_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
    print("Training benign model Finished!")

    benign_model = model_construct(args,args.test_model,data,device).to(device) 
    # In[7]:
    benign_ca = benign_model.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Benign CA: {:.4f}".format(benign_ca))

benign_model = benign_model.cpu()

# In[9]:
from sklearn_extra import cluster
from models.backdoor import Backdoor    #, defend_baseline_construct
from heuristic_selection import obtain_attach_nodes,cluster_distance_selection,cluster_degree_selection

from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
# here is randomly select poison nodes from unlabeled nodes
if(args.selection_method == 'none'):
    idx_attach = obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    idx_attach = cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)

# In[10]:
# train trigger generator 
model = Backdoor(args,device)
if(args.attack_method == 'Basic'):
    model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()
elif(args.attack_method == 'Rand_Gene' or args.attack_method == 'Rand_Samp'):
    model.fit_rand(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned_rand()
elif(args.attack_method == 'None'):
    train_edge_weights = torch.ones([train_edge_index.shape[1]],device=device,dtype=torch.float)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = data.x.clone(), train_edge_index.clone(), train_edge_weights, data.y.clone()
elif(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
    train_edge_weights = torch.ones([train_edge_index.shape[1]],device=device,dtype=torch.float)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = data.x.clone(), train_edge_index.clone(), train_edge_weights, data.y.clone()
    # build surrogate model
    surrogate_model = model_construct(args,args.model,data,device).to(device) 
    t_total = time.time()
    print("Training Surrogate Model...")
    surrogate_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training Completed...")
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
if(args.test_model == 'RobustGCN'):
    sp_poison_adj = SparseTensor.from_edge_index(poison_edge_index,poison_edge_weights,sparse_sizes = ([poison_x.shape[0],poison_x.shape[0]]))
    test_model.fit(poison_x, sp_poison_adj, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    output = test_model(poison_x,sp_poison_adj,poison_edge_weights)
else:
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    output = test_model(poison_x,poison_edge_index,poison_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate)) 

train_attach_rate = (poison_labels[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate)) 
#%%
from torch_sparse import SparseTensor
from baseline_atk import baseline_attack_parser, attack_baseline_construct
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
if(args.test_model=='RobustGCN'):
    sp_induct_adj = SparseTensor.from_edge_index(induct_edge_index,induct_edge_weights,sparse_sizes = ([poison_x.shape[0],poison_x.shape[0]]))
    clean_acc = test_model.test(poison_x,sp_induct_adj,induct_edge_weights,data.y,idx_clean_test)
else:
    clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
if(args.evaluate_mode == '1by1'):
    from torch_geometric.utils  import k_hop_subgraph
    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
    asr = 0
    for i, idx in enumerate(idx_atk):
        idx=int(idx)
        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
        ori_node_idx = sub_induct_nodeset[sub_mapping]
        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
        # inject trigger on attack test nodes (idx_atk)'''
        if(args.attack_method == 'Basic'):
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
        elif(args.attack_method == 'Rand_Gene' or args.attack_method == 'Rand_Samp'):
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger_rand(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,data.y)
        elif(args.attack_method == 'None'):
            induct_x, induct_edge_index,induct_edge_weights = poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights
        elif(args.attack_method == 'TDGIA'):
            induct_x, induct_edge_index,induct_edge_weights = poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights
            atk_param = baseline_attack_parser(args,device)
            atk_param['n_inject_max'] = 3
            atk_param['n_edge_max'] = 1
            attacker = attack_baseline_construct(args,atk_param)
            sp_induct_adj = SparseTensor(row=induct_edge_index[0], col=induct_edge_index[1], value=induct_edge_weights)
            induct_adj, x_attack = attacker.attack(model=surrogate_model,
                                                    adj=sp_induct_adj,
                                                    features=induct_x,
                                                    target_idx=torch.tensor([relabeled_node_idx]),
                                                    labels=None)
            induct_x = torch.cat([induct_x,x_attack],dim=0)
            induct_edge_index,induct_edge_weights = dense_to_sparse(induct_adj.to_dense())
        induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
        # # do pruning in test datas'''
        if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
            induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
        # attack evaluation
        if(args.test_model == 'RobustGCN'):
            sp_induct_adj = SparseTensor.from_edge_index(induct_edge_index,induct_edge_weights,sparse_sizes = ([induct_x.shape[0],induct_x.shape[0]]))
            output = test_model(induct_x,sp_induct_adj,induct_edge_weights)
            if(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
                train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]!=data.y[ori_node_idx]).float().mean()
            else:
                train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
            print("Node {}: {}".format(i, idx))
            print("ASR: {:.4f}".format(train_attach_rate))
        else:
            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
            if(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
                train_attach_rate = (output.argmax(dim=1)[idx_atk]!=data.y[idx_atk]).float().mean()
            else:
                train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
            print("Node {}: {}".format(i, idx))
            print("ASR: {:.4f}".format(train_attach_rate))
        asr += train_attach_rate
    asr = asr/(idx_atk.shape[0])
    print("Overall ASR: {:.4f}".format(asr))
elif(args.evaluate_mode == 'overall'):
    # %% inject trigger on attack test nodes (idx_atk)'''
    if(args.attack_method == 'Basic'):
        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    elif(args.attack_method == 'Rand_Gene' or args.attack_method == 'Rand_Samp'):
        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger_rand(idx_atk,poison_x,induct_edge_index,induct_edge_weights,data.y)
    elif(args.attack_method == 'None'):
        induct_x, induct_edge_index,induct_edge_weights = poison_x,induct_edge_index,induct_edge_weights
    elif(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
        from torch_geometric.utils import from_scipy_sparse_matrix
        induct_x, induct_edge_index,induct_edge_weights = poison_x,data.edge_index,induct_edge_weights
        induct_edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        atk_param = baseline_attack_parser(args,device)
        atk_param['n_inject_max'] = idx_atk.shape[0] * 3
        atk_param['n_edge_max'] = 1
        attacker = attack_baseline_construct(args,atk_param)
        sp_induct_adj = SparseTensor(row=induct_edge_index[0], col=induct_edge_index[1], value=induct_edge_weights)
        induct_adj, x_attack = attacker.attack(model=surrogate_model,
                                                adj=sp_induct_adj,
                                                features=induct_x,
                                                target_idx=idx_atk,
                                                labels=None)
        induct_x = torch.cat([induct_x.cpu(),x_attack.cpu()],dim=0).to(device)
        row,col,val = induct_adj.coo()
        induct_edge_index = torch.tensor(np.array([list(row.cpu()),list(col.cpu())]),device=device)
        induct_edge_weights = torch.ones([induct_edge_index.shape[1]],device=device,dtype=torch.float)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
    # do pruning in test datas'''
    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
    # attack evaluation
    if(args.test_model == 'RobustGCN'):
        sp_induct_adj = SparseTensor.from_edge_index(induct_edge_index,induct_edge_weights,sparse_sizes = ([induct_x.shape[0],induct_x.shape[0]]))
        output = test_model(induct_x,sp_induct_adj,induct_edge_weights)
        if(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
            train_attach_rate = (output.argmax(dim=1)[idx_atk]!=data.y[idx_atk]).float().mean()
        else:
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
        print("ASR: {:.4f}".format(train_attach_rate))
        ca = test_model.test(induct_x,sp_induct_adj,induct_edge_weights,data.y,idx_clean_test)
        print("CA: {:.4f}".format(ca))
    else:
        output = test_model(induct_x,induct_edge_index,induct_edge_weights)
        if(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
            train_attach_rate = (output.argmax(dim=1)[idx_atk]!=data.y[idx_atk]).float().mean()
        else:
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
        print("ASR: {:.4f}".format(train_attach_rate))
        ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        print("CA: {:.4f}".format(ca))