import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import Baseline_Attack.attacks.metric as metric
import Baseline_Attack.attacks.utils as utils
from Baseline_Attack.attacks.utils import EarlyStop


class PGD(object):
    r"""

    Description
    -----------
    Graph injection attack version of Projected Gradient Descent attack (`PGD <https://arxiv.org/abs/1706.06083>`__).

    Parameters
    ----------
    epsilon : float
        Perturbation level on features.
    n_epoch : int
        Epoch of perturbations.
    n_inject_max : int
        Maximum number of injected nodes.
    n_edge_max : int
        Maximum number of edges of injected nodes.
    feat_lim_min : float
        Minimum limit of features.
    feat_lim_max : float
        Maximum limit of features.
    loss : func of torch.nn.functional, optional
        Loss function compatible with ``torch.nn.functional``. Default: ``F.nll_loss``.
    eval_metric : func of grb.evaluator.metric, optional
        Evaluation metric. Default: ``metric.eval_acc``.
    device : str, optional
        Device used to host data. Default: ``cpu``.
    early_stop : bool, optional
        Whether to early stop. Default: ``False``.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.

    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 early_stop=False,
                 verbose=True):
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose

        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
        else:
            self.early_stop = early_stop

    def attack(self, model, adj, features, target_idx, labels=None):
        model.to(self.device)
        model.eval()
        n_total, n_feat = features.shape

        if labels == None:
            pred_orig = model(features, adj)
            origin_labels = torch.argmax(pred_orig, dim=1)
        else:
            origin_labels = labels.view(-1)
        
        adj_attack = self.injection(adj=utils.tensor_to_adj(adj),
                                    n_inject=self.n_inject_max,
                                    n_node=n_total,
                                    target_idx=target_idx)
        adj_attack = utils.adj_to_tensor(adj_attack).to(target_idx.device)
        # Random initialization
        features_attack = np.random.normal(loc=0, scale=self.feat_lim_max / 10,
                                           size=(self.n_inject_max, n_feat))

        features_attack = self.update_features(model=model,
                                               adj_attack=adj_attack,
                                               features=features,
                                               features_attack=features_attack,
                                               origin_labels=origin_labels,
                                               target_idx=target_idx)

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_node, target_idx):
        r"""

        Description
        -----------
        Randomly inject nodes to target nodes.

        Parameters
        ----------
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        n_inject : int
            Number of injection.
        n_node : int
            Number of all nodes.
        target_idx : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.

        """

        # test_index = torch.where(target_idx)[0]
        target_idx = target_idx.cpu()
        n_test = target_idx.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            islinked = np.zeros(n_test)
            for j in range(self.n_edge_max):
                x = i + n_node

                yy = random.randint(0, n_test - 1)
                while islinked[yy] > 0:
                    yy = random.randint(0, n_test - 1)

                islinked[yy] = 1
                y = target_idx[yy]
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])

        add1 = sp.csr_matrix((n_inject, n_node))
        add2 = sp.csr_matrix((n_node + n_inject, n_inject))
        adj_attack = sp.vstack([adj, add1])
        adj_attack = sp.hstack([adj_attack, add2])
        adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
        adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
        adj_attack.data = np.hstack([adj_attack.data, new_data])

        return adj_attack

    def update_features(self, model, adj_attack, features, features_attack, origin_labels, target_idx):
        r"""

        Description
        -----------
        Update features of injected nodes.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj_attack :  scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        features_attack : torch.FloatTensor
            Features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.
        origin_labels : torch.LongTensor
            Labels of target nodes originally predicted by the model.
        target_idx : torch.Tensor
            Mask of target nodes in form of ``N * 1`` torch bool tensor.
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.

        Returns
        -------
        features_attack : torch.FloatTensor
            Updated features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        epsilon = self.epsilon
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features.shape[0]

        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        model.eval()

        for i in range(n_epoch):
            features_attack.requires_grad_(True)
            features_attack.retain_grad()
            features_concat = torch.cat((features, features_attack), dim=0)
            pred = model(features_concat, adj_attack)

            # shall be pre_loss = +loss
            pred_loss = self.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx]).to(self.device)

            model.zero_grad()
            pred_loss.backward()
            grad = features_attack.grad.data
            features_attack = features_attack.clone() + epsilon * grad.sign()
            features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
            features_attack = features_attack.detach()

            test_score = self.eval_metric(pred[:n_total][target_idx],
                                          origin_labels[target_idx])

            if self.early_stop:
                self.early_stop(test_score)
                if self.early_stop.stop:
                    print("Attacking: Early stopped.")
                    self.early_stop.reset()
                    return features_attack

            if self.verbose:
                print(
                    "Attacking: Epoch {}, Loss: {:.5f}, Surrogate test score: {:.5f}".format(i, pred_loss, test_score),
                    end='\r' if i != n_epoch - 1 else '\n')

        return features_attack
