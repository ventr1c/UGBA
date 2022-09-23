from operator import add
import random
from Baseline_Attack.attacks import utils
import torch.nn.functional as F
import numpy as np
import torch
import scipy.sparse as sp
from Baseline_Attack.attacks.attack import gcn_norm


def dice_injection(adj, n_inject, n_edge_max, origin_labels, target_idx, device):
    n_classes = max(origin_labels)+1
    class_pos = [[] for i in range(n_classes)]
    for i in origin_labels:
        class_id = origin_labels[i]
        class_pos[class_id].append(i)
    direct_edges = n_edge_max//2    # number of edges connect to target nodes
    bridge_edges = n_edge_max-direct_edges  # number of edges connect to different classes

    n_node = adj.size(0)
    adj=utils.tensor_to_adj(adj)
    target_idx = target_idx.cpu()
    n_test = target_idx.shape[0]
    new_edges_x = []
    new_edges_y = []
    new_data = []

    # connect injected nodes to target nodes
    for i in range(n_inject):
        islinked = np.zeros(n_test)
        for j in range(direct_edges):
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
    adj_attack = utils.adj_to_tensor(adj_attack).to(device)
    return adj_attack

def random_class_injection(adj, n_inject, n_edge_max, origin_labels, target_idx, device, not_full=False):
    n_classes = max(origin_labels)+1
    class_pos = [[] for i in range(n_classes)]
    min_class_len = len(target_idx)
    for (i,pos) in enumerate(target_idx):
        class_id = origin_labels[pos]
        class_pos[class_id].append(i)
        
    for c in class_pos:
        min_class_len = min(min_class_len,len(class_pos[class_id]))

    if not not_full:
        assert min_class_len >= n_edge_max, print(f"min_class_len {min_class_len}")
    n_node = adj.size(0)
    adj=utils.tensor_to_adj(adj)
    target_idx = target_idx.cpu()
    n_test = target_idx.shape[0]
    new_edges_x = []
    new_edges_y = []
    new_data = []
    for i in range(n_inject):
        islinked = np.zeros(n_test)
        class_id = random.randint(0, n_classes-1)
        n_connections = min(len(class_pos[class_id]),n_edge_max)
        for j in range(n_connections):
            x = i + n_node

            yy = random.randint(0, len(class_pos[class_id]) - 1)
            while islinked[class_pos[class_id][yy]] > 0:
                yy = random.randint(0, len(class_pos[class_id]) - 1)
            
            islinked[class_pos[class_id][yy]] = 1
            y = target_idx[class_pos[class_id][yy]]
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
    adj_attack = utils.adj_to_tensor(adj_attack).to(device)
    return adj_attack


def random_injection(adj, n_inject, n_edge_max, target_idx, device):
    n_node = adj.size(0)
    adj=utils.tensor_to_adj(adj)
    target_idx = target_idx.cpu()
    n_test = target_idx.shape[0]
    new_edges_x = []
    new_edges_y = []
    new_data = []
    for i in range(n_inject):
        islinked = np.zeros(n_test)
        for j in range(n_edge_max):
            x = i + n_node
            yy = random.randint(0, n_test - 1)
            while islinked[yy] > 0:
                yy = random.randint(0, n_test - 1)
            
            # BUG: never duplicating linked nodes
            # solution
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
    adj_attack = utils.adj_to_tensor(adj_attack).to(device)
    return adj_attack


def tdgia_injection(adj, n_inject, n_edge_max, origin_labels, current_pred, 
            target_idx, device, self_connect_ratio=0.0, weight1=0.9, weight2=0.1):
    n_current = adj.size(0)
    adj=utils.tensor_to_adj(adj)
    target_idx = target_idx.cpu()
    n_test = target_idx.size(0)
    n_classes = origin_labels.max() + 1
    n_connect = int(n_edge_max * (1 - self_connect_ratio))
    n_self_connect = int(n_edge_max * self_connect_ratio)
    new_edges_x = []
    new_edges_y = []
    new_data = []
    add_score = np.zeros(n_test)
    deg = np.array(adj.sum(axis=0))[0] + 1.0
    for i in range(n_test):
        it = target_idx[i]
        label = origin_labels[it]
        score = current_pred[it][label] + 2
        add_score1 = score / deg[it]
        add_score2 = score / np.sqrt(deg[it])
        sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect + n_self_connect)
        add_score[i] = sc
    # higher score is better
    sorted_rank = add_score.argsort()
    sorted_rank = sorted_rank[-n_inject * n_connect:]
    labelgroup = np.zeros(n_classes)

    # separate them by origin_labels
    labelil = []
    for i in range(n_classes):
        labelil.append([])
    random.shuffle(sorted_rank)
    for i in sorted_rank:
        label = origin_labels[target_idx[i]]
        labelgroup[label] += 1
        labelil[label].append(i)
    pos = np.zeros(n_classes)
    for i in range(n_inject):
        for j in range(n_connect):
            smallest = 1
            small_id = 0
            for k in range(n_classes):
                if len(labelil[k]) > 0:
                    if (pos[k] / len(labelil[k])) < smallest:
                        smallest = pos[k] / len(labelil[k])
                        small_id = k
            # print((k,smallest))
            # if smallest == 1:
            #     for k in range(n_classes):
            #         print((pos[k],len(labelil[k])))
            #     print((len(target_idx),n_inject, n_edge_max))
            # print(labelil,small_id,pos[small_id])
            tu = labelil[small_id][int(pos[small_id])]
            pos[small_id] += 1
            x = n_current + i
            y = target_idx[tu]
            new_edges_x.extend([x, y])
            new_edges_y.extend([y, x])
            new_data.extend([1, 1])
    is_linked = np.zeros((n_inject, n_inject))
    for i in range(n_inject):
        rnd_times = 100
        while np.sum(is_linked[i]) < n_self_connect and rnd_times > 0:
            x = i + n_current
            rnd_times = 100
            yy = random.randint(0, n_inject - 1)
            while (np.sum(is_linked[yy]) >= n_self_connect or yy == i or
                   is_linked[i][yy] == 1) and (rnd_times > 0):
                yy = random.randint(0, n_inject - 1)
                rnd_times -= 1
            if rnd_times > 0:
                y = n_current + yy
                is_linked[i][yy] = 1
                is_linked[yy][i] = 1
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])
    add1 = sp.csr_matrix((n_inject, n_current))
    add2 = sp.csr_matrix((n_current + n_inject, n_inject))
    adj_attack = sp.vstack([adj, add1])
    adj_attack = sp.hstack([adj_attack, add2])
    adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
    adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
    adj_attack.data = np.hstack([adj_attack.data, new_data])
    adj_attack = utils.adj_to_tensor(adj_attack).to(device)
    return adj_attack

def atdgia_injection(adj, n_inject, n_edge_max, origin_labels, current_pred, 
            target_idx, device, self_connect_ratio=0.0, weight1=0.9, weight2=0.1):
    n_current = adj.size(0)
    adj=utils.tensor_to_adj(adj)
    target_idx = target_idx.cpu()
    n_test = target_idx.size(0)
    n_classes = origin_labels.max() + 1
    n_connect = int(n_edge_max * (1 - self_connect_ratio))
    n_self_connect = int(n_edge_max * self_connect_ratio)
    new_edges_x = []
    new_edges_y = []
    new_data = []
    add_score = np.zeros(n_test)
    deg = np.array(adj.sum(axis=0))[0] + 1.0
    for i in range(n_test):
        it = target_idx[i]
        label = origin_labels[it]
        cur_label = current_pred[it].argmax()
        if cur_label==label:
            score = 1.0 - current_pred[it][label]
        else:
            score = 0
        # score = current_pred[it][label] + 2
        add_score1 = score / deg[it]
        add_score2 = score / np.sqrt(deg[it])
        sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect + n_self_connect)
        add_score[i] = sc
    # higher score is better
    sorted_rank = add_score.argsort()
    sorted_rank = sorted_rank[-n_inject * n_connect:]
    labelgroup = np.zeros(n_classes)

    # separate them by origin_labels
    labelil = []
    for i in range(n_classes):
        labelil.append([])
    random.shuffle(sorted_rank)
    for i in sorted_rank:
        label = origin_labels[target_idx[i]]
        labelgroup[label] += 1
        labelil[label].append(i)
    pos = np.zeros(n_classes)
    for i in range(n_inject):
        for j in range(n_connect):
            smallest = 1
            small_id = 0
            for k in range(n_classes):
                if len(labelil[k]) > 0:
                    if (pos[k] / len(labelil[k])) < smallest:
                        smallest = pos[k] / len(labelil[k])
                        small_id = k
            # print((k,smallest))
            # if smallest == 1:
            #     for k in range(n_classes):
            #         print((pos[k],len(labelil[k])))
            #     print((len(target_idx),n_inject, n_edge_max))
            tu = labelil[small_id][int(pos[small_id])]
            pos[small_id] += 1
            x = n_current + i
            y = target_idx[tu]
            new_edges_x.extend([x, y])
            new_edges_y.extend([y, x])
            new_data.extend([1, 1])
    is_linked = np.zeros((n_inject, n_inject))
    for i in range(n_inject):
        rnd_times = 100
        while np.sum(is_linked[i]) < n_self_connect and rnd_times > 0:
            x = i + n_current
            rnd_times = 100
            yy = random.randint(0, n_inject - 1)
            while (np.sum(is_linked[yy]) >= n_self_connect or yy == i or
                   is_linked[i][yy] == 1) and (rnd_times > 0):
                yy = random.randint(0, n_inject - 1)
                rnd_times -= 1
            if rnd_times > 0:
                y = n_current + yy
                is_linked[i][yy] = 1
                is_linked[yy][i] = 1
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])
    add1 = sp.csr_matrix((n_inject, n_current))
    add2 = sp.csr_matrix((n_current + n_inject, n_inject))
    adj_attack = sp.vstack([adj, add1])
    adj_attack = sp.hstack([adj_attack, add2])
    adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
    adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
    adj_attack.data = np.hstack([adj_attack.data, new_data])
    adj_attack = utils.adj_to_tensor(adj_attack).to(device)
    return adj_attack


def tdgia_ranking_select(adj, n_inject, n_edge_max, origin_labels, current_pred, target_idx, ratio=0.5, neg=False, weight1=0.9, weight2=0.1):
    # ranking with tdgia vulnerability score
    target_idx = target_idx.cpu()
    n_test = target_idx.size(0)
    n_connect = n_edge_max


    add_score = np.zeros(n_test)
    deg = np.array(adj.sum(-1).long().cpu()) + 1.0
    for i in range(n_test):
        it = target_idx[i]
        label = origin_labels[it]
        cur_label = current_pred[it].argmax()
        score = current_pred[it][label] + 2
        add_score1 = score / deg[it]
        add_score2 = score / np.sqrt(deg[it])
        sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect)
        add_score[i] = sc
    
    if neg:
        add_score = -add_score
    sorted_rank = add_score.argsort()

    sel_len = int(len(target_idx)*ratio)
    sorted_rank = sorted_rank[-sel_len:]
    sel_idx = target_idx[sorted_rank]
    
    return sel_idx

def atdgia_ranking_select(adj, n_inject, n_edge_max, origin_labels, current_pred, target_idx, ratio=0.5, neg=False, weight1=0.9, weight2=0.1):
    # ranking with tdgia+ vulnerability score

    target_idx = target_idx.cpu()
    n_test = target_idx.size(0)
    n_connect = n_edge_max


    add_score = np.zeros(n_test)
    deg = np.array(adj.sum(-1).long().cpu()) + 1.0
    for i in range(n_test):
        it = target_idx[i]
        label = origin_labels[it]
        cur_label = current_pred[it].argmax()
        if cur_label==label:
            score = 1.0 - current_pred[it][label]
        else:
            score = 0
        # score = current_pred[it][label] + 2
        add_score1 = score / deg[it]
        add_score2 = score / np.sqrt(deg[it])
        sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect)
        add_score[i] = sc
    
    if neg:
        add_score = -add_score
    sorted_rank = add_score.argsort()

    sel_len = int(len(target_idx)*ratio)
    sorted_rank = sorted_rank[-sel_len:]
    sel_idx = target_idx[sorted_rank]
    
    return sel_idx


def tdgia_class_injection(adj, n_inject, n_edge_max, origin_labels, current_pred, 
            target_idx, device, weight1=0.9, weight2=0.1):
    n_current = adj.size(0)
    adj=utils.tensor_to_adj(adj)
    target_idx = target_idx.cpu()
    n_test = target_idx.size(0)
    n_classes = origin_labels.max() + 1
    n_connect = n_edge_max

    new_edges_x = []
    new_edges_y = []
    new_data = []
    add_score = np.zeros(n_test)
    deg = np.array(adj.sum(axis=0))[0] + 1.0
    for i in range(n_test):
        it = target_idx[i]
        label = origin_labels[it]
        score = current_pred[it][label] + 2
        add_score1 = score / deg[it]
        add_score2 = score / np.sqrt(deg[it])
        sc = weight1 * add_score1 + weight2 * add_score2 / np.sqrt(n_connect + n_self_connect)
        add_score[i] = sc
    # higher score is better
    sorted_rank = add_score.argsort()
    sorted_rank = sorted_rank[-n_inject * n_connect:]
    labelgroup = np.zeros(n_classes)

    # separate them by origin_labels
    labelil = []
    for i in range(n_classes):
        labelil.append([])
    random.shuffle(sorted_rank)
    for i in sorted_rank:
        label = origin_labels[target_idx[i]]
        labelgroup[label] += 1
        labelil[label].append(i)
    pos = np.zeros(n_classes)
    for i in range(n_inject):
        for j in range(n_connect):
            smallest = 1
            small_id = 0
            for k in range(n_classes):
                if len(labelil[k]) > 0:
                    if (pos[k] / len(labelil[k])) < smallest:
                        smallest = pos[k] / len(labelil[k])
                        small_id = k
            # print((k,smallest))
            # if smallest == 1:
            #     for k in range(n_classes):
            #         print((pos[k],len(labelil[k])))
            #     print((len(target_idx),n_inject, n_edge_max))
            tu = labelil[small_id][int(pos[small_id])]
            pos[small_id] += 1
            x = n_current + i
            y = target_idx[tu]
            new_edges_x.extend([x, y])
            new_edges_y.extend([y, x])
            new_data.extend([1, 1])
    is_linked = np.zeros((n_inject, n_inject))
    for i in range(n_inject):
        rnd_times = 100
        while np.sum(is_linked[i]) < n_self_connect and rnd_times > 0:
            x = i + n_current
            rnd_times = 100
            yy = random.randint(0, n_inject - 1)
            while (np.sum(is_linked[yy]) >= n_self_connect or yy == i or
                   is_linked[i][yy] == 1) and (rnd_times > 0):
                yy = random.randint(0, n_inject - 1)
                rnd_times -= 1
            if rnd_times > 0:
                y = n_current + yy
                is_linked[i][yy] = 1
                is_linked[yy][i] = 1
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])
    add1 = sp.csr_matrix((n_inject, n_current))
    add2 = sp.csr_matrix((n_current + n_inject, n_inject))
    adj_attack = sp.vstack([adj, add1])
    adj_attack = sp.hstack([adj_attack, add2])
    adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
    adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
    adj_attack.data = np.hstack([adj_attack.data, new_data])
    adj_attack = utils.adj_to_tensor(adj_attack).to(device)
    return adj_attack

from torch_sparse import SparseTensor


def agia_injection(attacker, model, adj, n_inject, n_edge_max, features, features_attack, target_idx, origin_labels, 
                    device, optim="adam", old_reg=False, real_target_idx=None, homophily=None):
    model.to(attacker.device)
    model.eval()
    n_epoch = attacker.n_epoch
    n_total = features.size(0)
    device = attacker.device
    epsilon = attacker.epsilon
    # setup the edge entries for optimization
    new_x = torch.cat([torch.LongTensor([i+n_total]).repeat(target_idx.size(0))
                      for i in range(n_inject)]).to(device)
    new_y = target_idx.repeat(n_inject).to(device)
    assert new_x.size() == new_y.size()
    vals = torch.zeros(new_x.size(0)).to(device)
    print(f"#original edges {adj.nnz()}, #target idx {len(target_idx)}, #init edges {vals.size(0)}")
    # jointly update adjecency matrix & features
    if adj.size(0)>n_total:
        print("init edge weights from the previous results")
        # that's a attacked adj
        orig_adj = adj[:n_total,:n_total]
        x, y, z = orig_adj.coo()
        # now we init val with the attacked graph
        vals[:] = 0
        x_inj, y_inj, _ = adj[n_total:,:].coo()
        idx_map = {}
        for (i, idx) in enumerate(target_idx):
            idx_map[idx.item()] = i 
        for i in range(n_inject*n_edge_max):
            xx, yy = x_inj[i], y_inj[i]
            pos = xx*len(target_idx)+idx_map[yy.cpu().item()]
            vals[pos] = 1
        old_vals = vals.clone()
    else:
        old_vals = None
        x, y, z = adj.coo()
    
    z = torch.ones(x.size(0)).to(device) if z == None else z
    isolate_idx = torch.nonzero((adj.sum(-1)==0)[:n_total].long(),as_tuple=True)[0].cpu()

    makeup_x = []
    makeup_y = []
    makeup_z = []
    for iidx in isolate_idx:
        makeup_x.append(iidx)
        makeup_y.append(iidx)
        makeup_z.append(1)
    x = torch.cat((x,torch.LongTensor(makeup_x).to(device)),dim=0)
    y = torch.cat((y,torch.LongTensor(makeup_y).to(device)),dim=0)
    z = torch.cat((z,torch.LongTensor(makeup_z).to(device)),dim=0)
    print(f"add self-con for {len(isolate_idx)} nodes")
    new_row = torch.cat((x, new_x, new_y), dim=0)
    new_col = torch.cat((y, new_y, new_x), dim=0)
    vals.requires_grad_(True)
    vals.retain_grad()
    adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))
    
    if optim == "adam":
        optimizer_adj = torch.optim.Adam([vals],epsilon)
    features_concat = torch.cat((features, features_attack), dim=0)
    old_layer_output = None
    orig_layer_output = None

    real_target_idx = target_idx[target_idx<origin_labels.size(0)] if real_target_idx==None else real_target_idx
    
    beta = 0.01 if n_edge_max >= 100 else 1
    for i in range(attacker.a_epoch):
        pred = model(features_concat, adj_attack)
       
        pred_loss = attacker.loss(pred[:n_total][real_target_idx],
                              origin_labels[real_target_idx])
        
        # sparsity loss for the adjacency matrix, based on L1 norm
        if optim=="adam" and not model.use_ln:
            sparsity_loss = beta*(n_edge_max*n_inject-torch.norm(vals,p=1))
        else:
            sparsity_loss = -0.01*abs(vals.view(n_inject,-1).sum(-1)-n_edge_max).mean()

        pred_loss = -pred_loss-sparsity_loss
        
        if optim == "adam":
            optimizer_adj.zero_grad()
        pred_loss.backward(retain_graph=True)

        if optim == "adam":
            optimizer_adj.step()
        else:
            grad = vals.grad.data
            vals = vals.detach() - epsilon * grad.sign()
            vals = torch.clamp(vals,0,1)
            vals.requires_grad_(True)
            vals.retain_grad()
            adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))


        test_score = attacker.eval_metric(pred[:n_total][real_target_idx],
                                      origin_labels[real_target_idx])
        if attacker.verbose:
            print("Attacking Edges: Epoch {}, Loss: {:.2f}, Surrogate test score: {:.2f}, injected edge {:}".format(
                    i, pred_loss, test_score, vals[:len(target_idx)].sum()),end='\r' if i != n_epoch - 1 else '\n')


    # select edges with higher weights as the final injection matrix
    tmp_vals = -vals.detach().view(n_inject, -1)
    sel_idx = tmp_vals.argsort(dim=-1)[:, :n_edge_max]
    sel_mask = torch.zeros(tmp_vals.size()).bool()
    for i in range(sel_idx.size(0)):
        sel_mask[i, sel_idx[i]] = True
    sel_idx = torch.nonzero(sel_mask.view(-1)).squeeze()

    new_x = new_x[sel_idx]
    new_y = new_y[sel_idx]
    print(f"Finally injected edges {len(new_x)}, minimum vals {vals[sel_idx].min()}, maximum vals {vals[sel_idx].max()}")

    new_row = torch.cat((x, new_x, new_y), dim=0)
    new_col = torch.cat((y, new_y, new_x), dim=0)
    adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.ones(new_row.size(0),device=device))
    if old_vals!=None:
        new_vals = torch.zeros(old_vals.size()).to(old_vals.device)
        new_vals[sel_idx] = 1
        print(f"number of modifications: {(old_vals-new_vals).abs().sum()}")
        print(f"added: {((-old_vals+new_vals)>0).sum()}")
        print(f"removed: {((old_vals-new_vals)>0).sum()}")
    return adj_attack



def meta_injection(attacker, model, adj, n_inject, n_edge_max, features, features_attack, target_idx, origin_labels, 
                    device, real_target_idx=None, homophily=None):
    model.to(device)
    model.eval()
    
    n_total = features.size(0)
    # setup the edge entries for optimization
    new_x = torch.cat([torch.LongTensor([i+n_total]).repeat(target_idx.size(0))
                      for i in range(n_inject)]).to(device)
    new_y = target_idx.repeat(n_inject).to(device)
    assert new_x.size() == new_y.size()
    vals = torch.zeros(new_x.size(0)).to(device)
    print(f"#original edges {adj.nnz()}, #target idx {len(target_idx)}, #init edges {vals.size(0)}")
    
    # jointly update adjecency matrix & features
    if adj.size(0)>n_total:
        print("init edge weights from the previous results")
        # that's a attacked adj
        orig_adj = adj[:n_total,:n_total]
        x, y, z = orig_adj.coo()
        # now we init val with the attacked graph
        vals[:] = 0
        x_inj, y_inj, _ = adj[n_total:,:].coo()
        idx_map = {}
        for (i, idx) in enumerate(target_idx):
            idx_map[idx.item()] = i 
        for (xx,yy) in zip(x_inj,y_inj):
            pos = xx*len(target_idx)+idx_map[yy.cpu().item()]
            vals[pos] = 1
    else:
        x, y, z = adj.coo()
    z = torch.ones(x.size(0)).to(device) if z == None else z
    new_row = torch.cat((x, new_x, new_y), dim=0)
    new_col = torch.cat((y, new_y, new_x), dim=0)
    vals.requires_grad_(True)
    adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))

    real_target_idx = target_idx[target_idx<origin_labels.size(0)] if real_target_idx==None else real_target_idx


    features_concat = torch.cat((features, features_attack), dim=0)
    pred = model(features_concat, adj_attack)
    pred_loss = attacker.loss(pred[:n_total][real_target_idx],
                            origin_labels[real_target_idx]).to(device)
    adj_meta_grad = torch.autograd.grad(pred_loss, vals, retain_graph=True)[0]
    # select the edges with largest meta gradient and flip
    vals = vals.detach().long()
    grad_order = adj_meta_grad.abs().argsort(descending=True)
    
    cnt_flip = 0    # count the number of flipped edges
    cnt_left = attacker.step_size
    flip_grad_sum = 0
    for pos in grad_order:
        if vals[pos] == 0 and adj_meta_grad[pos]>0:
            vals[pos] = 1
            cnt_flip += 1
            cnt_left -= 1
            flip_grad_sum += adj_meta_grad[pos]
        elif vals[pos]>0 and adj_meta_grad[pos]<0:
            vals[pos] = 0
            cnt_flip -= 1
            cnt_left -= 1
            flip_grad_sum += adj_meta_grad[pos].item()
        if cnt_left == 0:
            break
    # flip the lowest
    low_grad_sum = 0
    grad_order = adj_meta_grad.argsort(descending=False)
    for pos in grad_order:
        if cnt_flip > 0 and vals[pos]==1:
            cnt_flip -= 1
            vals[pos] = 0
            low_grad_sum += adj_meta_grad[pos]
        elif cnt_flip < 0 and vals[pos]==0:
            vals[pos] = 1
            cnt_flip += 1
            low_grad_sum += adj_meta_grad[pos]
        if cnt_flip == 0:
            break
    if low_grad_sum>=flip_grad_sum:
        print("No upd onto edges")
        return adj_attack.detach()
    vals = vals.bool() 
    new_x = new_x[vals]
    new_y = new_y[vals]
    print(f"Finally injected edges {len(new_x)}")
    new_row = torch.cat((x, new_x, new_y), dim=0)
    new_col = torch.cat((y, new_y, new_x), dim=0)
    adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.ones(new_row.size(0),device=device))
    pred = model(features_concat, adj_attack)
    pred_loss_new = attacker.loss(pred[:n_total][real_target_idx],
                            origin_labels[real_target_idx]).to(device)
    print(f"loss gain {pred_loss_new-pred_loss}")
    return adj_attack
