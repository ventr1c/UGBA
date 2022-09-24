
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.GCN_Encoder import GCN_Encoder
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork,Reddit,Reddit2,Flickr,Yelp,PPI
from torch_geometric.utils import to_dense_adj,dense_to_sparse

# import ogb

# from ogb import nodeproppred
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import DataLoader

from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,select_target_nodes
import help_funcs
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
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
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=200, help='Number of epochs to train trigger generator.')
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
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.15,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=0,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--attack_method', type=str, default='None',
                    choices=['Rand_Gene','Rand_Samp','Basic','None'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--trigger_prob', type=float, default=0.5,
                    help="The probability to generate the trigger's edges in random method")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='overall',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
# device2 = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id+1))


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)

#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

# np.random.seed(11) # fix the random seed is important
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
from models.backdoor import model_construct
benign_modelpath = './modelpath/{}_{}_benign.pth'.format(args.model, args.dataset)
if(os.path.exists(benign_modelpath)):
    # load existing benign model
    benign_model = torch.load(benign_modelpath, map_location=device).to(device)
    # benign_model = benign_model.to(device)
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
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
benign_model = benign_model.cpu()

# In[9]:

from sklearn_extra import cluster
from models.backdoor import Backdoor
from heuristic_selection import cluster_degree_selection_seperate_fixed, obtain_attach_nodes, obtain_attach_nodes_by_cluster_degree, obtain_attach_nodes_by_cluster_gpu,obtain_attach_nodes_by_influential,obtain_attach_nodes_by_cluster,cluster_distance_selection,cluster_degree_selection, cluster_degree_selection_seperate

from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
# here is randomly select poison nodes from unlabeled nodes
if(args.selection_method == 'none'):
    idx_attach = obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'loss' or args.selection_method == 'conf'):
    idx_attach = obtain_attach_nodes_by_influential(args,benign_model,unlabeled_idx.cpu().tolist(),data.x,train_edge_index,None,data.y,device,size,selected_way=args.selection_method)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster'):
    idx_attach = cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    # idx_attach = cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    # idx_attach = cluster_degree_selection_seperate(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
# In[10]:
# train trigger generator 
model = Backdoor(args,device)
if(args.attack_method == 'Basic'):
    model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()
elif(args.attack_method == 'Rand_Gene' or args.attack_method == 'Rand_Samp'):
    # model.fit_rand(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned_rand(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
elif(args.attack_method == 'None'):
    train_edge_weights = torch.ones([train_edge_index.shape[1]],device=device,dtype=torch.float)
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = data.x.clone(), train_edge_index.clone(), train_edge_weights, data.y.clone()
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
# if(args.attack_method == 'None'):
#     bkd_tn_nodes = idx_train
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
#%%
test_model = model_construct(args,args.test_model,data,device).to(device) 
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

output = test_model(poison_x,poison_edge_index,poison_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))
# torch.cuda.empty_cache()
#%%
induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
# test_model = test_model.cpu()

print("accuracy on clean test nodes: {:.4f}".format(clean_acc))

# poison_x, poison_edge_index, poison_edge_weights, poison_labels = poison_x.to(device2), poison_edge_index.to(device2), poison_edge_weights.to(device2), poison_labels.to(device2)
# model.trojan = model.trojan.cpu()
if(args.evaluate_mode == '1by1'):
    from torch_geometric.utils  import k_hop_subgraph
    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
    asr = 0
    flip_asr = 0
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
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
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger_rand(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,data.y[sub_induct_nodeset], full_data=False)
        elif(args.attack_method == 'None'):
            induct_x, induct_edge_index,induct_edge_weights = poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights

        induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
        # # do pruning in test datas'''
        if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
            induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
        # attack evaluation

        # test_model = test_model.to(device)
        output = test_model(induct_x,induct_edge_index,induct_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
        print("Node {}: {}, Origin Label: {}".format(i, idx, data.y[idx]))
        print("ASR: {:.4f}".format(train_attach_rate))
        asr += train_attach_rate
        if(data.y[idx] != args.target_class):
            flip_asr += train_attach_rate
        # ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        # print("CA: {:.4f}".format(ca))
    asr = asr/(idx_atk.shape[0])
    flip_asr = flip_asr/(flip_idx_atk.shape[0])
    print("Overall ASR: {:.4f}".format(asr))
    print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
elif(args.evaluate_mode == 'overall'):
    # %% inject trigger on attack test nodes (idx_atk)'''
    if(args.attack_method == 'Basic'):
        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    elif(args.attack_method == 'Rand_Gene' or args.attack_method == 'Rand_Samp'):
        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger_rand(idx_atk,poison_x,induct_edge_index,induct_edge_weights,data.y)
    elif(args.attack_method == 'None'):
        induct_x, induct_edge_index,induct_edge_weights = poison_x,induct_edge_index,induct_edge_weights

    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
    # do pruning in test datas'''
    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
    # attack evaluation

    # test_model = test_model.to(device)
    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    print("ASR: {:.4f}".format(train_attach_rate))
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
    print("CA: {:.4f}".format(ca))