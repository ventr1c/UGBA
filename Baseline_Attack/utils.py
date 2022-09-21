import os
import numpy as np
from typing import List
from torch_sparse import SparseTensor
from tqdm import tqdm as core_tqdm
import torch
import random
import codecs

import math
import random
import torch
from torch_geometric.utils import to_undirected, k_hop_subgraph
import torch_geometric.transforms as T


def prune_graph(adj_test, target_idx, k):
    adj_test = adj_test.cpu()
    u, v, _ = adj_test.coo()
    _, edge_index, __, ___ = k_hop_subgraph(target_idx,k,torch.stack((u,v),dim=0))
    graph_size = torch.Size((adj_test.size(0),adj_test.size(1)))
    new_adj_test = SparseTensor(row=edge_index[1], col=edge_index[0], value=None, sparse_sizes=graph_size,is_sorted=True).to_symmetric()
    return new_adj_test

def target_select(model,adj,features,labels,target_idx,num):
    # num highest margin
    # num lowest margin
    # 2num random
    # with torch.no_grad():
    #     pred = model(features,adj)[target_idx]
    #     pred_y = pred.argmax(-1)
    # correct_idx = labels[target_idx].view(-1)==pred_y.view(-1)
    # assert len(correct_idx) >= 4*num
    # pred_max = pred.max(-1)[0]
    # second_y =  pred
    # second_y[torch.arange(pred_y.size(0)),pred_y] = -1e9
    # margin = (pred_max-second_y.max(-1)[0])[correct_idx]
    # margin_max = margin.argsort()
    # random_ids = torch.randperm(len(margin)-2*num)[:2*num]
    # selected_ids = torch.cat((margin_max[:num],margin_max[-num:],margin_max[num:-num][random_ids]),dim=0)


    # sanity check
    with torch.no_grad():
        pred = model(features,adj)[target_idx]
        pred_y = pred.argmax(-1)
    pred_sort, _ = pred.sort(-1,descending=True)
    correct_idx = labels[target_idx].view(-1)==pred_y.view(-1)
    print(f"Correctly classified nodes: {correct_idx.sum()}")
    new_margin = pred_sort[correct_idx,0]-pred_sort[correct_idx,1]
    new_margin_max = new_margin.argsort()
    random_ids = torch.randperm(len(new_margin)-2*num)[:2*num]
    # (min, max, random, random)
    selected_ids = torch.cat((new_margin_max[:num],new_margin_max[-num:],new_margin_max[num:-num][random_ids]),dim=0)

    # assert (new_margin_max[:num]!=margin_max[:num]).sum()==0, print((new_margin_max[:num]!=margin_max[:num]).sum()) 
    # assert (new_margin_max[-num:]!=margin_max[-num:]).sum()==0, print((new_margin_max[:num]!=margin_max[:num]).sum()) 

    return target_idx[selected_ids]

def feat_normalize(features, norm=None, lim_min=-1.0, lim_max=1.0):
    r"""
    Description
    -----------
    Feature normalization function.

    Parameters
    ----------
    features : torch.FloatTensor
        Features in form of ``N * D`` torch float tensor.
    norm : str, optional
        Type of normalization. Choose from ``["linearize", "arctan", "tanh", "standarize"]``.
        Default: ``None``.
    lim_min : float
        Minimum limit of feature value. Default: ``-1.0``.
    lim_max : float
        Minimum limit of feature value. Default: ``1.0``.

    Returns
    -------
    features : torch.FloatTensor
        Normalized features in form of ``N * D`` torch float tensor.

    """
    if norm == "linearize":
        k = (lim_max - lim_min) / (features.max() - features.min())
        features = lim_min + k * (features - features.min())
    elif norm == "arctan":
        features = (features - features.mean()) / features.std()
        features = 2 * np.arctan(features) / np.pi
    elif norm == "tanh":
        features = (features - features.mean()) / features.std()
        features = np.tanh(features)
    elif norm == "standardize":
        features = (features - features.mean()) / features.std()
    else:
        features = features

    return features

def train_test_split_edges(data, use_mask=False, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        train_mask (bool, optional): if it's True, we will sample edges 
            accoding to the pre-defined split. (default: :`False`)
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.
    
    if use_mask:
        # only use edges from trainset
        new_data = T.ToSparseTensor()(data)
        adj_train = new_data.adj_t[data.train_mask][:,data.train_mask]
        tval_mask = torch.logical_or(data.train_mask,data.val_mask)
        adj_val = new_data.adj_t[tval_mask][:,tval_mask]
        row, col = adj_val.coo()[:2]
        num_nodes = sum(tval_mask).item()
        print(f"# of edges for training: {len(row)}")
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = random.sample(range(neg_row.size(0)), min(n_v + n_t,
                                                     neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def get_index_induc(index_a, index_b):
    r"""

    Description
    -----------
    Get index under the inductive training setting.

    Parameters
    ----------
    index_a : tuple
        Tuple of index.
    index_b : tuple
        Tuple of index.

    Returns
    -------
    index_a_new : tuple
        Tuple of mapped index.
    index_b_new : tuple
        Tuple of mapped index.

    """

    i_a, i_b = 0, 0
    l_a, l_b = len(index_a), len(index_b)
    i_new = 0
    index_a_new, index_b_new = [], []
    while i_new < l_a + l_b:
        if i_a == l_a:
            while i_b < l_b:
                i_b += 1
                index_b_new.append(i_new)
                i_new += 1
            continue
        elif i_b == l_b:
            while i_a < l_a:
                i_a += 1
                index_a_new.append(i_new)
                i_new += 1
            continue
        if index_a[i_a] < index_b[i_b]:
            i_a += 1
            index_a_new.append(i_new)
            i_new += 1
        else:
            i_b += 1
            index_b_new.append(i_new)
            i_new += 1

    return index_a_new, index_b_new

def inductive_split(adj, split_idx):
    """
    inductive split adjs for PyG graphs
    will automatically use relative ids for splitted graphs
    """
    adj_train = adj[split_idx["train"]][:,split_idx["train"]]
    train_mask = torch.zeros(adj.size(0)).bool()
    train_mask[split_idx["train"]] = 1
    val_mask = torch.zeros(adj.size(0)).bool()
    val_mask[split_idx["valid"]] = 1
    train_val_mask = torch.logical_or(train_mask, val_mask)
    adj_val = adj[train_val_mask][:,train_val_mask]
    adj_test = adj
    return adj_train, adj_val, adj_test


def set_rand_seed(rand_seed):
    rand_seed = rand_seed if rand_seed >= 0 else torch.initial_seed() % 4294967295  # 2^32-1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)


def extra_misg_ids(args, model, data, train_idx):
    """
    sample misclassified training samples
    and save to args.misg_path
    """
    y_true = data.y
    assert len(args.misg_path)>0
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.adj_t)[train_idx]
        y_pred = out.argmax(dim=-1).to(y_true.device)
        model.train() 
    misg_ids = torch.nonzero(y_pred!=y_true[train_idx].view(-1),as_tuple=True)[0]
    misg_ids = train_idx[misg_ids].cpu()
    assert len(np.intersect1d(misg_ids,train_idx.cpu()))==len(misg_ids)
    misclass_data = {"ids":misg_ids,"preds":y_pred,"labels":y_true[train_idx]}
    misg_path = os.path.join(args.misg_path,"_".join([args.dataset,args.model]))
    print(f"Saving misclassified data to {misg_path+'.pt'}")
    print(f"Saving the trained GNN to {misg_path+'.model'}")
    torch.save(misclass_data,misg_path+'.pt')
    torch.save(model.state_dict(),misg_path+'.model')


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def load_np_embedding(path: str):
    embedding = np.load(path)

    return embedding

def save_np_embedding(path: str, embedding: np.ndarray):
    path_dir = os.sep.join(path.split(os.sep)[:-1])
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    np.save(path,embedding)

def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


class tqdm(core_tqdm):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def load_prebuilt_word_embedding(embedding_path, embedding_dim):
    """
    Read prebuilt word embeddings from a file
    :param embedding_path: string, file path of the word embeddings
    :param embedding_dim: int, dimensionality of the word embeddings
    :return: a dictionary mapping each word to its corresponding word embeddings
    """
    word_embedding_map = dict()

    if embedding_path is not None and len(embedding_path) > 0:
        for line in codecs.open(embedding_path, mode="r", encoding="utf-8"):
            line = line.strip()
            if not line or len(line.split())<=2:
                continue
            else:
                word_embedding = line.split()
                # print(word_embedding)
                assert len(word_embedding) == 1 + embedding_dim, print(len(word_embedding))
                word = word_embedding[0]
                embedding = [float(val) for val in word_embedding[1:]]
                if word in word_embedding_map.keys():
                    continue
                else:
                    word_embedding_map[word] = embedding
    # print(len(word_embedding_map.keys()),sorted(word_embedding_map.keys()))
    sorted_prebuilt_words = np.zeros((len(word_embedding_map.keys()),embedding_dim))
    for i in range(len(word_embedding_map.keys())):
        sorted_prebuilt_words[i] = word_embedding_map[str(i)]
    return sorted_prebuilt_words

import pickle as pkl
import sys

import networkx as nx
import scipy.sparse as sp

# geom-gcn
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
