import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import torch
import scipy.sparse as sp

def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in edge_index:
        sims.append(float(F.cosine_similarity(features[u].unsqueeze(0),features[v].unsqueeze(0))))
    sims = np.array(sims)
    # print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")
    return sims

def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    # update structure
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    return updated_edge_index,updated_edge_weights

def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        # calculate edge simlarity
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    edge_weights[dissim_edges_index] = 0
    # select the nodes between dissimilar edgesy
    dissim_edges = edge_index[:,dissim_edges_index]    # output: [[v_1,v_2],[u_1,u_2]]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))
    # update structure
    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    return updated_edge_index,updated_edge_weights,dissim_nodes 

def select_target_nodes(args,seed,model,features,edge_index,edge_weights,labels,idx_val,idx_test):
    test_ca,test_correct_index = model.test_with_correct_nodes(features,edge_index,edge_weights,labels,idx_test)
    test_correct_index = test_correct_index.tolist()
    '''select target test nodes'''
    test_correct_nodes = idx_test[test_correct_index].tolist()
    # filter out the test nodes that are not in target class
    target_class_nodes_test = [int(nid) for nid in idx_test
            if labels[nid]==args.target_class] 
    # get the target test nodes
    idx_val,idx_test = idx_val.tolist(),idx_test.tolist()
    rs = np.random.RandomState(seed)
    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))  # the test nodes not in target class is candidate atk_test_nodes
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num)
    '''select clean test nodes'''
    cand_clean_test_nodes = list(set(idx_test) - set(atk_test_nodes))
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num)
    '''select poisoning nodes from unlabeled nodes (assign labels is easier than change, also we can try to select from labeled nodes)'''
    N = features.shape[0]
    cand_poi_train_nodes = list(set(idx_val)-set(atk_test_nodes)-set(clean_test_nodes))
    poison_nodes_num = int(N * args.vs_ratio)
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num)
    
    return atk_test_nodes, clean_test_nodes,poi_train_nodes

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()