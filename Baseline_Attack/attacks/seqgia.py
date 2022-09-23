from Baseline_Attack.attacks.attack import deg_estimate, edge_sim_estimate, gcn_norm, gia_update_features, init_feat, node_sim_estimate, smooth_update_features
from Baseline_Attack.attacks.injection import agia_injection, meta_injection, random_class_injection, random_injection, tdgia_injection, tdgia_ranking_select, atdgia_injection, atdgia_ranking_select
from Baseline_Attack.utils import feat_normalize
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import Baseline_Attack.attacks.metric as metric
import Baseline_Attack.attacks.utils as utils
from Baseline_Attack.attacks.utils import EarlyStop
    
class SEQGIA(object):
    r"""

    Graph Injection Attack in a Squential manner

    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 a_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 early_stop=0,
                 verbose=True,
                 disguise_coe=1.0,
                 sequential_step=0.2,
                 injection="random",
                 feat_upd="gia",
                 branching=False,
                 iter_epoch=2,
                 agia_pre=0.5,
                 hinge=False):
        self.sequential_step = sequential_step
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.a_epoch = a_epoch
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
            self.early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
        else:
            self.early_stop = early_stop
        self.branching = branching
        self.injection = injection.lower()
        self.feat_upd = feat_upd.lower()
        self.iter_epoch = iter_epoch
        self.agia_pre = agia_pre
        self.hinge=hinge

    def attack(self, model, adj, features, target_idx, labels=None):
        model.to(self.device)
        model.eval()
        n_total, n_feat = features.shape

        if labels == None:
            pred_orig = model(features, adj)
            origin_labels = torch.argmax(pred_orig, dim=1)
        else:
            origin_labels = labels.view(-1)
        

        self.adj_degs = torch.zeros((self.n_inject_max,)).long()+self.n_edge_max
        features_h = node_sim_estimate(features,adj,self.n_inject_max)
        n_inject_total = 0
        adj_attack = adj
        features_attack = None
        tot_target_nodes = len(target_idx)
        """
        Sequential injection
        """
        while n_inject_total < self.n_inject_max:
            
            if n_inject_total>0:
                with torch.no_grad():
                    current_pred = F.softmax(model(torch.cat((features,features_attack),dim=0), adj_attack), dim=1)
            else:
                current_pred = pred_orig
            n_inject_cur = min(self.n_inject_max-n_inject_total,max(1,int(self.n_inject_max * self.sequential_step)))
            n_target_cur = min(tot_target_nodes,max(n_inject_cur*(self.n_edge_max+1),int(tot_target_nodes * self.sequential_step)))
            if self.branching:
                cur_target_idx = atdgia_ranking_select(adj, n_inject_cur, self.n_edge_max, origin_labels, current_pred, target_idx, ratio=n_target_cur/len(target_idx))
            else:
                cur_target_idx = target_idx
            print("Attacking: Sequential inject {}/{} nodes, target {}/{} nodes".format(n_inject_total + n_inject_cur, self.n_inject_max,len(cur_target_idx),len(target_idx)))
            if self.injection == "tdgia":
                adj_attack = tdgia_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, current_pred, cur_target_idx, self.device)
            elif self.injection == "atdgia":
                adj_attack = atdgia_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, current_pred, cur_target_idx, self.device)
            elif self.injection == "class":
                adj_attack = random_class_injection(adj_attack, n_inject_cur, self.n_edge_max, origin_labels, cur_target_idx, self.device)
            elif self.injection == "meta":
                self.step_size = self.n_edge_max
                features_tmp = torch.cat((features,features_attack),dim=0) if features_attack!=None else features
                adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                meta_epoch = max(1,(n_inject_cur//6)*1) if self.n_inject_max <=600 else (n_inject_cur//60)*10
                for i in range(meta_epoch):
                    features_attack_new = init_feat(n_inject_cur, features, self.device, style="random", 
                                            feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
                    features_attack_new = gia_update_features(self, model, adj_attack, features_tmp, features_attack_new, origin_labels, target_idx, 
                                            features_h[n_inject_total:n_inject_total+n_inject_cur],hinge=self.hinge)
                    adj_attack = meta_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, real_target_idx=target_idx, homophily=features_h)
            elif self.injection[-4:] == "agia":
                if (n_inject_total+n_inject_cur) < int(self.n_inject_max*self.agia_pre):
                    adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                else:
                    if self.injection[0].lower() == "a":
                        # the default approach
                        self.opt = "adam"
                        self.old_reg = False
                    else:
                        raise Exception("Not implemented")
                    
                    features_tmp = torch.cat((features,features_attack),dim=0) if features_attack!=None else features
                    for epoch in range(self.iter_epoch):
                        if epoch == 0:
                            adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)
                        else:
                            adj_attack = agia_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, self.opt, old_reg=False, real_target_idx=target_idx, homophily=features_h)
                            if self.old_reg:
                                adj_attack = agia_injection(self, model, adj_attack, n_inject_cur, self.n_edge_max, features_tmp, 
                                            features_attack_new, cur_target_idx, origin_labels, self.device, self.opt, old_reg=True, real_target_idx=target_idx, homophily=features_h)
                            
                        features_attack_new = init_feat(n_inject_cur, features, self.device, style="random", 
                                        feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
                        features_attack_new = gia_update_features(self, model, adj_attack, features_tmp, features_attack_new, origin_labels, target_idx, 
                                                                homophily=features_h[n_inject_total:n_inject_total+n_inject_cur],hinge=self.hinge)
            else:
                adj_attack = random_injection(adj_attack, n_inject_cur, self.n_edge_max, cur_target_idx, self.device)

            features_attack_new = init_feat(n_inject_cur, features, self.device, style="normal", 
                                        feat_lim_min=self.feat_lim_min, feat_lim_max=self.feat_lim_max)
            features_attack = torch.cat((features_attack,features_attack_new),dim=0) if features_attack!=None else features_attack_new
            
            if self.injection.lower() in ["tdgia","atdgia"]:
                features_attack = smooth_update_features(self, model, adj_attack, features, features_attack, origin_labels, target_idx, 
                                                        features_h[:n_inject_total+n_inject_cur],n_inject_cur,hinge=self.hinge)
            else:
                features_attack = gia_update_features(self, model, adj_attack, features, features_attack, origin_labels, target_idx, 
                                                        features_h[:n_inject_total+n_inject_cur],hinge=self.hinge)
            n_inject_total += n_inject_cur
        return adj_attack, features_attack
