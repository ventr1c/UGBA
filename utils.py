#%%
import torch
import numpy as np

def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.
    Parameters
    ----------
    labels : torch.LongTensor
        node labels
    Returns
    -------
    torch.LongTensor
        onehot labels tensor
    """
    labels = labels.long()
    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask
import scipy.sparse as sp
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
# %%
def subgraph(subset,edge_index, edge_attr = None, relabel_nodes: bool = False):
    """Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # if relabel_nodes:
    #     node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
    #                            device=device)
    #     node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    #     edge_index = node_idx[edge_index]


    return edge_index, edge_attr, edge_mask
# %%

def get_split(args,data, device):
    rs = np.random.RandomState(10)
    perm = rs.permutation(data.num_nodes)
    train_number = int(0.2*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(0.2*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]

    return data, idx_train, idx_val, idx_clean_test, idx_atk