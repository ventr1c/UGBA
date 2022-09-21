from Baseline_Attack.attacks.attack import deg_estimate, edge_sim_estimate, gcn_norm, gia_update_features, init_feat, node_sim_estimate
from Baseline_Attack.attacks.injection import random_injection
from Baseline_Attack.utils import feat_normalize
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import Baseline_Attack.attacks.metric as metric
import Baseline_Attack.attacks.utils as utils
from Baseline_Attack.attacks.utils import EarlyStop


class GIA(object):
    r"""

    Node similarity regularized PGD on graphs

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
                 verbose=True,
                 disguise_coe=1.0,
                 hinge=False):
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
        self.disguise_coe = disguise_coe
        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=1000, epsilon=1e-4)
        else:
            self.early_stop = early_stop
        self.hinge=  hinge

    def attack(self, model, adj, features, target_idx, labels=None):
        model.to(self.device)
        model.eval()
        n_total, n_feat = features.shape

        if labels == None:
            pred_orig = model(features, adj)
            origin_labels = torch.argmax(pred_orig, dim=1)
        else:
            origin_labels = labels.view(-1)
        
        # self.adj_degs = deg_estimate(adj,self.n_inject_max)
        self.adj_degs = torch.zeros((self.n_inject_max,)).long()+self.n_edge_max
        adj_attack = random_injection(adj,self.n_inject_max, self.n_edge_max, target_idx, self.device)
        # Random initialization
        features_attack = init_feat(self.n_inject_max, features, self.device, style="random", 
                                    feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
        features_h = node_sim_estimate(features,adj,self.n_inject_max)
        # self.edges_h = edge_sim_estimate(features,adj,self.n_inject_max*self.n_edge_max)
        # features_h = node_sim_estimate(features,adj,features_attack.shape[0],style='random')
        features_attack = gia_update_features(self,model,adj_attack,features,features_attack,origin_labels,target_idx,features_h,hinge=self.hinge)

        return adj_attack, features_attack
