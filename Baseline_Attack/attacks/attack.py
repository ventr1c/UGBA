import random
from Baseline_Attack.attacks import utils
import torch.nn.functional as F
import numpy as np
import torch
import scipy.sparse as sp


def init_feat(num, features, device, style="sample", feat_lim_min=-1, feat_lim_max=1):
    if style.lower() == "sample":
        # do random sample from features to init x
        feat_len = features.size(0)
        x = torch.empty((num,features.size(1)),device=features.device)
        sel_idx = torch.randint(0,feat_len,(num,1))
        x = features[sel_idx.view(-1)].clone()
    elif style.lower() == "normal":
        x = torch.randn((num,features.size(1))).to(features.device)
    elif style.lower() == "zeros":
        x = torch.zeros((num,features.size(1))).to(features.device)
    else:
        x = np.random.normal(loc=0, scale=feat_lim_max/10, size=(num, features.size(1)))
        x = utils.feat_preprocess(features=x, device=device)
    return x

# edge-centric cosine similarity analysis
def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in zip(edge_index[0],edge_index[1]):
        sims.append(F.cosine_similarity(features[u].unsqueeze(0),
                                        features[v].unsqueeze(0)).cpu().numpy())
    sims = np.array(sims)
    return sims


def edge_sim_estimate(x, adj, num, style='sample'):
    """
    estimate the mean and variance from the observed data points
    """
    edge_index = adj.coo()[:2]
    sims = edge_sim_analysis(edge_index, x)
    if style.lower() == 'random':
        hs = np.random.choice(sims,size=(num,))
        hs = torch.FloatTensor(hs).to(x.device)
    else:
        mean, var = sims.mean(), sims.var()
        hs = torch.randn((num,)).to(x.device)
        hs = mean + hs*torch.pow(torch.tensor(var),0.5)
    return hs


# node-centric cosine similarity analysis
# analyze 1-hop neighbor cosine similarity
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=None)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

def node_sim_analysis(adj, x):
    adj = gcn_norm(adj,add_self_loops=False)
    x_neg = adj @ x
    node_sims = F.cosine_similarity(x_neg,x).cpu().numpy()
    return node_sims


def node_sim_estimate(x, adj, num, style='sample'):
    """
    estimate the mean and variance from the observed data points
    """
    sims = node_sim_analysis(adj,x)
    if style.lower() == 'random':
        hs = np.random.choice(sims,size=(num,))
        hs = torch.FloatTensor(hs).to(x.device)
    else:
        # mean, var = sims.mean(), sims.var()
        # hs = torch.randn((num,)).to(x.device)
        # hs = mean + hs*torch.pow(torch.tensor(var),0.5)
        from scipy.stats import skewnorm
        a, loc, scale = skewnorm.fit(sims)
        hs = skewnorm(a, loc, scale).rvs(num)
        hs = torch.FloatTensor(hs).to(x.device)
    return hs


def deg_estimate(adj, num, style='sample'):
    degs = adj.sum(1).cpu().numpy()
    if style.lower() == 'random':
        degs_est = np.random.choice(degs,size=(num,))
        degs_est = torch.LongTensor(degs_est)
    else:
        import powerlaw
        dist = powerlaw.Fit(degs,discrete=True)
        new_deg_pl = dist.power_law.generate_random(min(num,1000))
        new_deg_pl -= new_deg_pl.min()-degs.mean()
        degs_est = torch.LongTensor(new_deg_pl[:num])
        
    return degs_est



# pgd feature upd
def update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, n_epoch=499, dist="cos"):
    attacker.early_stop.reset()
    if hasattr(attacker, 'disguise_coe'):
        disguise_coe = attacker.disguise_coe
    else:
        disguise_coe = 0

    epsilon = attacker.epsilon
    n_epoch = min(n_epoch,attacker.n_epoch)
    feat_lim_min, feat_lim_max = attacker.feat_lim_min, attacker.feat_lim_max
    n_total = features.shape[0]
    if dist.lower()=="cos":
        dis = lambda x: F.cosine_similarity(x[0],x[1])
    elif dist.lower() == "l2":
        dis = lambda x: F.pairwise_distance(x[0],x[1],p=2)
    else:
        raise Exception(f"no implementation for {dist}")
    
    features_attack = utils.feat_preprocess(features=features_attack, device=attacker.device)
    model.eval()

    attack_degs = torch.unique(adj_attack.coo()[1],return_counts=True)[1][-features_attack.size(0):]
    for i in range(n_epoch):
        features_attack.requires_grad_(True)
        features_attack.retain_grad()
        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attack)
        # stablize the pred_loss, only if disguise_coe > 0
        weights = pred[target_idx,origin_labels[target_idx]].exp()>=min(disguise_coe,1e-8)
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx],reduction='none')
        pred_loss = (pred_loss*weights).mean()
        
        # (inversely) maximize the differences 
        # between attacked features and neighbors' features
        with torch.no_grad():
            features_propagate = adj_attack @ torch.cat((features,torch.zeros(features_attack.size()).to(features.device)),dim=0) 
            features_propagate = features_propagate[n_total:]/attack_degs.unsqueeze(1)#expand(features_attack.size())

            
        homo_loss = disguise_coe*dis((features_attack, features_propagate)).mean()
        
        pred_loss += homo_loss
        model.zero_grad()
        pred_loss.backward()
        grad = features_attack.grad.data
        features_attack = features_attack.clone() + epsilon * grad.sign()
        features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
        features_attack = features_attack.detach()
        test_score = attacker.eval_metric(pred[:n_total][target_idx],
                                      origin_labels[target_idx])
        if attacker.early_stop:
            attacker.early_stop(test_score)
            if attacker.early_stop.stop:
                print("Attacking: Early stopped.")
                attacker.early_stop.reset()
                return features_attack
        if attacker.verbose:
            print(
                "Attacking: Epoch {}, Loss: {:.5f}, Surrogate test score: {:.5f}".format(i, pred_loss, test_score),
                end='\r' if i != n_epoch - 1 else '\n')
    return features_attack

# gia feature upd
def gia_update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, homophily=None, hinge=False):
    attacker.early_stop.reset()
    if hasattr(attacker, 'disguise_coe'):
        disguise_coe = attacker.disguise_coe
    else:
        disguise_coe = 0
    epsilon = attacker.epsilon
    n_epoch = attacker.n_epoch
    feat_lim_min, feat_lim_max = attacker.feat_lim_min, attacker.feat_lim_max
    n_total = features.shape[0]
    model.eval()

    features_propagate = None
    for i in range(n_epoch):
        features_attack.requires_grad_(True)
        features_attack.retain_grad()
        features_concat = torch.cat((features, features_attack), dim=0)
        pred = model(features_concat, adj_attack)
        # stablize the pred_loss, only if disguise_coe > 0
        weights = pred[target_idx,origin_labels[target_idx]].exp()>=min(disguise_coe,1e-8)
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx],reduction='none')
        pred_loss = (pred_loss*weights).mean()

        if features_propagate == None:
            # (inversely) maximize the differences 
            # between attacked features and neighbors' features
            with torch.no_grad():
                features_propagate = gcn_norm(adj_attack, add_self_loops=False) @ features_concat
                features_propagate = features_propagate[n_total:]
        sims = F.cosine_similarity(features_attack, features_propagate)
        if homophily!=None:
            # minimize the distance to sampled homophily
            if hinge:
                # hinge loss
                mask = sims < homophily
                new_disguise_coe = torch.ones(sims.size(),device=sims.device)
                new_disguise_coe[mask] = disguise_coe
                new_disguise_coe[torch.logical_not(mask)] = disguise_coe*0.5
                homo_loss = (new_disguise_coe * (sims - homophily)).mean()
            else:
                homo_loss = disguise_coe * ((sims - homophily).mean())
        else:
            # maximize similarity
            homo_loss = disguise_coe*sims.mean()

        pred_loss += homo_loss
        model.zero_grad()
        pred_loss.backward()
        grad = features_attack.grad.data
        features_attack = features_attack.detach() + epsilon * grad.sign()
        features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
        test_score = attacker.eval_metric(pred[:n_total][target_idx],
                                      origin_labels[target_idx])
        if attacker.early_stop:
            attacker.early_stop(test_score)
            if attacker.early_stop.stop:
                print("Attacking: Early stopped.")
                attacker.early_stop.reset()
                return features_attack
        if attacker.verbose:
            print(
                "Attacking: Epoch {}, Loss: {:.5f}, Surrogate test score: {:.5f}".format(i, pred_loss, test_score),
                end='\r' if i != n_epoch - 1 else '\n')
    return features_attack



# smooth feature upd from tdgia
def smooth_update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, homophily=None, n_inject_cur=0, hinge=False):
    if hasattr(attacker, 'disguise_coe'):
        disguise_coe = attacker.disguise_coe
    else:
        disguise_coe = 0
    epsilon = attacker.epsilon
    n_epoch = attacker.n_epoch
    feat_lim_min, feat_lim_max = attacker.feat_lim_min, attacker.feat_lim_max
    n_total = features.shape[0]
    model.eval()

    features_attack = features_attack.cpu().data.numpy()
    features_attack = features_attack / feat_lim_max
    features_attack[:-n_inject_cur] = np.arcsin(features_attack[:-n_inject_cur])
    features_attack = utils.feat_preprocess(features=features_attack, device=attacker.device)

    features_attack.requires_grad_(True)
    optimizer = torch.optim.Adam([features_attack],lr=epsilon)
    
    features_propagate = None
    for i in range(n_epoch):
        features_attack_sin = torch.sin(features_attack) * feat_lim_max    
        features_concat = torch.cat((features, features_attack_sin), dim=0)

        pred = model(features_concat, adj_attack)
        
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                               origin_labels[target_idx],reduction="none")
        
        if features_propagate == None:
            # (inversely) maximize the differences 
            # between attacked features and neighbors' features
            with torch.no_grad():
                features_propagate = gcn_norm(adj_attack, add_self_loops=False) @ features_concat
                features_propagate = features_propagate[n_total:]
        sims = F.cosine_similarity(features_attack_sin, features_propagate)
        if homophily!=None:
            # minimize the distance to sampled homophily
            if hinge:
                # hinge loss
                mask = sims < homophily
                # print(mask.sum())
                new_disguise_coe = torch.ones(sims.size(),device=sims.device)
                new_disguise_coe[mask] = disguise_coe
                new_disguise_coe[torch.logical_not(mask)] = disguise_coe*0.5
                homo_loss = (new_disguise_coe * (sims - homophily)).mean()
            else:
                homo_loss = disguise_coe * ((sims - homophily).mean())
        else:
            # maximize similarity
            homo_loss = disguise_coe*sims.mean()

        pred_loss += homo_loss
        pred_loss = F.relu(-pred_loss + 5) ** 2
        pred_loss = pred_loss.mean()
        optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        optimizer.step()
        test_score = attacker.eval_metric(pred[:n_total][target_idx],
                                      origin_labels[target_idx])
        if attacker.early_stop:
            attacker.early_stop(test_score)
            if attacker.early_stop.stop:
                print("Attacking: Early stopped.")
                attacker.early_stop.reset()
                return features_attack_sin.detach()
        if attacker.verbose:
            print(
                "Attacking: Epoch {}, Loss: {:.5f}, Surrogate test score: {:.5f}".format(i, pred_loss, test_score),
                end='\r' if i != n_epoch - 1 else '\n')

    return features_attack_sin.detach()
