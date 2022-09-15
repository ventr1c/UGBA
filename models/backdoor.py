#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
from models.GCN_Encoder import GCN_Encoder
def model_construct(args,model_name,data,device):
    if (model_name == 'GCN'):
        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device)
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(data.y.max()+1), 
                    heads=8,
                    dropout=args.dropout, 
                    lr=args.train_lr, 
                    weight_decay=args.weight_decay, 
                    device=device)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.train_lr,\
                weight_decay=args.weight_decay,\
                device=device)
    elif(model_name == 'GCN_Encoder'):
        model = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=args.hidden,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=args.dropout,                    
                            lr=args.train_lr,                    
                            weight_decay=args.weight_decay,                    
                            device=device)
    return model

#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
import torch.optim as optim
from models.GCN import GCN

def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def obtain_attach_nodes(node_idxs, size):
    ### current random to implement
    size = min(len(node_idxs),size)
    return node_idxs[np.random.choice(len(node_idxs),size,replace=False)]

def obtain_attach_nodes_by_influential(args,model,node_idxs,x,edge_index,edge_weights,labels,device,size,selected_way='conf'):
    size = min(len(node_idxs),size)
    # return node_idxs[np.random.choice(len(node_idxs),size,replace=False)]
    loss_fn = F.nll_loss
    model = model.to(device)
    labels = labels.to(device)
    model.eval()
    output = model(x, edge_index, edge_weights)
    loss_diffs = []
    '''select based on the diff between the loss on target class and true class, nodes with larger diffs are easily selected '''
    if(selected_way == 'loss'):
        candidate_nodes = np.array([])

        for id in range(output.shape[0]):
            loss_atk = loss_fn(output[id],torch.LongTensor([args.target_class]).to(device)[0])
            loss_bef = loss_fn(output[id],labels[id])
            # print(loss_atk,loss_bef)
            loss_diff = float(loss_atk - loss_bef)
            loss_diffs.append(loss_diff)
        loss_diffs = np.array(loss_diffs)

        # split the nodes according to the label
        label_list = np.unique(labels.cpu())
        labels_dict = {}
        for i in label_list:
            labels_dict[i] = np.where(labels.cpu()==i)[0]
            # filter out labeled nodes
            labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))
        # fairly select from all the class except for the target class 
        each_selected_num = int(size/len(label_list)-1)
        last_seleced_num = size - each_selected_num*(len(label_list)-2)
        for label in label_list:
            single_labels_nodes = labels_dict[label]    # the node idx of the nodes in single class
            single_labels_nodes = np.array(list(set(single_labels_nodes)))
            single_labels_nodes_loss = loss_diffs[single_labels_nodes]
            single_labels_nid_index = np.argsort(-single_labels_nodes_loss) # sort descently based on the loss
            sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
            if(label != label_list[-1]):
                candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
            else:
                candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
        return candidate_nodes.astype(int)
    elif(selected_way == 'conf'):
        '''select based on the diff between the conf on target class and true class, nodes with larger confidents are easily selected '''
        candidate_nodes = np.array([])
        confidences = []
        # calculate the confident of each node
        output = model(x, edge_index, edge_weights)
        softmax = torch.nn.Softmax(dim=1)
        for i in range(output.shape[0]):
            output_nids = output[[i]]
            preds = output_nids.max(1)[1].type_as(labels)
            preds = preds.cpu()
            correct = preds.eq(labels[[i]].detach().cpu()).double().sum().item()
            confidence = torch.mean(torch.max(softmax(output_nids), dim=1)[0]).item()
            confidences.append(confidence)
        confidences = np.array(confidences)
        # split the nodes according to the label
        label_list = np.unique(labels.cpu())
        labels_dict = {}
        for i in label_list:
            labels_dict[i] = np.where(labels.cpu()==i)[0]
            labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))
        # fairly select from all the class except for the target class 
        each_selected_num = int(size/len(label_list)-1)
        last_seleced_num = size - each_selected_num*(len(label_list)-2)
        for label in label_list:
            single_labels_nodes = labels_dict[label]
            single_labels_nodes = np.array(list(set(single_labels_nodes)))
            single_labels_nodes_conf = confidences[single_labels_nodes]
            single_labels_nid_index = np.argsort(-single_labels_nodes_conf)
            sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
            if(label != label_list[-1]):
                candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
            else:
                candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
        return candidate_nodes.astype(int)

def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    
def obtain_attach_nodes_by_cluster(args,model,node_idxs,x,labels,device,size):
    dis_weight = args.dis_weight
    cluster_centers = model.cluster_centers_
    y_pred = model.predict(x.detach().cpu().numpy())
    # y_true = labels.cpu().numpy()
    # calculate the distance of each nodes away from their centers
    distances = [] 
    distances_tar = []
    for id in range(x.shape[0]):
        # tmp_center_label = args.target_class
        tmp_center_label = y_pred[id]
        # tmp_true_label = y_true[id]
        tmp_tar_label = args.target_class
        
        tmp_center_x = cluster_centers[tmp_center_label]
        # tmp_true_x = cluster_centers[tmp_true_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        # dis1 = np.linalg.norm(tmp_true_x - x[id].cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        # print(dis,dis1,tmp_center_label,tmp_true_label)
        distances.append(dis)
        distances_tar.append(dis_tar)
        
    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    # label_list = np.unique(labels.cpu())
    print(y_pred)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        # labels_dict[i] = np.where(labels.cpu()==i)[0]
        labels_dict[i] = np.where(y_pred==i)[0]
        # filter out labeled nodes
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size/len(label_list)-1)
    last_seleced_num = size - each_selected_num*(len(label_list)-2)
    candidate_nodes = np.array([])
    for label in label_list:
        if(label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]    # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))

        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)
        # the closer to the center, the more far away from the target centers
        single_labels_dis_score = dis_weight * single_labels_nodes_dis + (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(single_labels_dis_score) # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if(label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes

def obtain_attach_nodes_by_cluster_gpu(args,y_pred,cluster_centers,node_idxs,x,labels,device,size):
    dis_weight = args.dis_weight
    # cluster_centers = model.cluster_centers_
    # y_pred = model.predict(x.detach().cpu().numpy())
    # y_true = labels.cpu().numpy()
    # calculate the distance of each nodes away from their centers
    distances = [] 
    distances_tar = []
    for id in range(x.shape[0]):
        # tmp_center_label = args.target_class
        tmp_center_label = y_pred[id]
        # tmp_true_label = y_true[id]
        tmp_tar_label = args.target_class
        
        tmp_center_x = cluster_centers[tmp_center_label]
        # tmp_true_x = cluster_centers[tmp_true_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        # dis1 = np.linalg.norm(tmp_true_x - x[id].cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        # print(dis,dis1,tmp_center_label,tmp_true_label)
        distances.append(dis)
        distances_tar.append(dis_tar)
        
    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    # label_list = np.unique(labels.cpu())
    print(y_pred)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        # labels_dict[i] = np.where(labels.cpu()==i)[0]
        labels_dict[i] = np.where(y_pred==i)[0]
        # filter out labeled nodes
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size/len(label_list)-1)
    last_seleced_num = size - each_selected_num*(len(label_list)-2)
    candidate_nodes = np.array([])
    for label in label_list:
        if(label == args.target_class):
            continue
        single_labels_nodes = labels_dict[label]    # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))

        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)
        # the closer to the center, the more far away from the target centers
        single_labels_dis_score =  single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(single_labels_dis_score) # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if(label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes


from torch_geometric.utils import to_undirected,erdos_renyi_graph
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.poison_x = None
        self.poison_edge_index = None
        self.poison_edge_weights = None

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*trigger_size])
            for j in range(trigger_size):
                for k in range(j):
                    edge_list.append([start+i*trigger_size+j,start+i*trigger_size+k])
        
        edge_index = torch.tensor(edge_list).long().T

        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):

                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat])

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()

            
            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            
            # involve unlabeled nodes in outter optimization
            self.trojan.eval()
            optimizer_trigger.zero_grad()

            idx_outter = torch.cat([idx_attach,idx_unlabeled[np.random.choice(len(idx_unlabeled),size=512,replace=False)]])

            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate
        
            trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()

            trojan_feat = trojan_feat.view([-1,features.shape[1]])

            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)

            update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
            update_feat = torch.cat([features,trojan_feat])
            update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class
            loss_target = self.args.target_loss_weight *F.nll_loss(output[torch.cat([idx_train,idx_outter])],
                                    labels_outter[torch.cat([idx_train,idx_outter])])
            loss_homo = 0.0

            if(self.args.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                            trojan_weights,\
                                            update_feat,\
                                            self.args.homo_boost_thrd)
            
            loss_outter = loss_target + self.args.homo_loss_weight * loss_homo

            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()
            

            if args.debug and i % 10 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo))
                print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))

        self.trojan.eval()
        torch.cuda.empty_cache()

    def inject_trigger_rand(self, idx_attach, features,edge_index,edge_weight, labels):

        if(self.args.attack_method == 'Rand_Gene'):
            trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_gene(features, self.idx_labeled, idx_attach)
        elif(self.args.attack_method == 'Rand_Samp'):
            trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_samp(features, self.idx_labeled, labels, idx_attach)
        update_edge_index = torch.cat([edge_index,trojan_edge_index],dim=1)
        update_edge_weights = torch.cat([edge_weight,trojan_weights]) 
        update_feat = torch.cat([features,trojan_feat])

        return update_feat, update_edge_index, update_edge_weights

    def fit_rand(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.idx_labeled = torch.concat([idx_train,idx_unlabeled]).to(self.device)
        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        
        self.homo_loss = HomoLoss(self.args,self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # change the labels of the poisoned node to the target class
        self.origin_labels = labels.clone()
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        # trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        # poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        
        for i in range(args.trojan_epochs):
            for j in range(self.args.inner):
                optimizer_shadow.zero_grad()
                if(self.args.attack_method == 'Rand_Gene'):
                    trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_gene(features, self.idx_labeled, idx_attach)
                elif(self.args.attack_method == 'Rand_Samp'):
                    trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_samp(features, self.idx_labeled, self.origin_labels, idx_attach)
 
                # trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                # trojan_weights = trojan_weights.flatten()
                # trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_index = torch.cat([edge_index,trojan_edge_index],dim=1)
                poison_edge_weights = torch.cat([edge_weight,trojan_weights]) 
                poison_x = torch.cat([features,trojan_feat])

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()

            
            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            
            # involve unlabeled nodes in outter optimization

            idx_outter = torch.cat([idx_attach,idx_unlabeled[np.random.choice(len(idx_unlabeled),size=512,replace=False)]])

            if(self.args.attack_method == 'Rand_Gene'):
                trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_gene(features, self.idx_labeled, idx_outter)
            elif(self.args.attack_method == 'Rand_Samp'):
                trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_samp(features, self.idx_labeled, labels, idx_outter)
        
            update_edge_index = torch.cat([edge_index,trojan_edge_index],dim=1)
            update_edge_weights = torch.cat([edge_weight,trojan_weights]) 
            update_feat = torch.cat([features,trojan_feat])

            # trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)

            # update_edge_weights = torch.cat([edge_weight,trojan_weights])
            # update_feat = torch.cat([features,trojan_feat])
            # update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class
            loss_target = 0.0 *F.nll_loss(output[torch.cat([idx_train,idx_outter])],
                                    labels_outter[torch.cat([idx_train,idx_outter])])
            loss_homo = 0.0

            if(self.args.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge_index,\
                                            trojan_weights,\
                                            update_feat,\
                                            self.args.homo_boost_thrd)
            
            loss_outter = loss_target + self.args.homo_loss_weight * loss_homo

            loss_outter.backward()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()
            

            if args.debug and i % 10 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo))
                print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))


    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
    
    def get_poisoned_rand(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger_rand(self.idx_attach,self.features,self.edge_index,self.edge_weights, self.origin_labels)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

    def trigger_rand_gene(self, x, idx_labeled, idx_attach):
        '''random generate'''
        '''
        construct connection: randomly select 
        '''
        nattach = len(idx_attach)
        # nedge = self.args.trigger_size*(self.args.trigger_size-1)
        # trojan_edge_weights = torch.zeros([nattach*nedge]) 
        start = x.shape[0]
        # trojan_edge_index = None
        # attach idx_attach with one node of the triggers 
        edge_list = []
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*self.args.trigger_size])
            # for j in range(self.args.trigger_size):
            #     for k in range(j):
            #         edge_list.append([start+i*self.args.trigger_size+j,start+i*self.args.trigger_size+k])
        # to undirected
        trojan_edge_index = torch.tensor(edge_list).long().T
        row = torch.cat([trojan_edge_index[0], trojan_edge_index[1]])
        col = torch.cat([trojan_edge_index[1],trojan_edge_index[0]])
        trojan_edge_index = torch.stack([row,col]).to(self.device)
        for i in range(nattach):
            tmp_edge_index = erdos_renyi_graph(self.args.trigger_size,edge_prob=self.args.trigger_prob).to(self.device)
            tmp_edge_index[0] = start + i*self.args.trigger_size + tmp_edge_index[0]
            tmp_edge_index[1] = start + i*self.args.trigger_size + tmp_edge_index[1]
            # if(trojan_edge_index == None):
            #     trojan_edge_index = torch.cat([tmp_edge_index[0].unsqueeze(0),tmp_edge_index[1].unsqueeze(0)],0)
            # else:
            trigger_row1_index = torch.cat([trojan_edge_index[0],tmp_edge_index[0]])
            trigger_row2_index = torch.cat([trojan_edge_index[1],tmp_edge_index[1]])
            trojan_edge_index = torch.cat([trigger_row1_index.unsqueeze(0),trigger_row2_index.unsqueeze(0)],0)
        trojan_edge_weights = torch.ones([trojan_edge_index.shape[1]],device=self.device,dtype=torch.float)
        # trojan_edge_weights = torch.zeros([nedge])
        # for i in range(nattach):
        #     for i in range(trigger_edge_index.shape[1]):
        #         row = trigger_edge_index[0][i]
        #         col = trigger_edge_index[1][i]
        #         # trojan_edge_weights[i*col+row*self.args.trigger_size] = 1
        #         trojan_edge_weights[i*nattach + ] = 1
            
        '''
        calculate the average of node features
        '''
        nodeNum = x[idx_labeled].shape[0]
        featDim = x[idx_labeled].shape[1]
        aver_featNum = int(len(x[idx_labeled].nonzero())/nodeNum)
        nattach = len(idx_attach)
        '''
        construct features: randomly sample N dims in one feature vectors and assign their items as 1
        '''
        injTriggerNum = nattach*self.args.trigger_size
        trojan_feat = torch.zeros((injTriggerNum,featDim)).to(self.device)
        rs = np.random.RandomState(self.args.seed)
        seeds = list(rs.choice(10000,size = injTriggerNum))
        for i in range(injTriggerNum):
            rs = np.random.RandomState(seeds[i])
            trojan_feat_dim_index = rs.choice(featDim, size = aver_featNum, replace=False)
            trojan_feat[i][trojan_feat_dim_index] = 1 
        
        return trojan_feat, trojan_edge_index, trojan_edge_weights

    def trigger_rand_samp(self, x, seen_node_idx, origin_labels, idx_attach):
        '''random sampling'''
        '''
        construct connection: randomly select 
        '''
        nattach = len(idx_attach)
        start = x.shape[0]

        edge_list = []
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*self.args.trigger_size])
        # to undirected
        trojan_edge_index = torch.tensor(edge_list).long().T
        row = torch.cat([trojan_edge_index[0], trojan_edge_index[1]])
        col = torch.cat([trojan_edge_index[1],trojan_edge_index[0]])
        trojan_edge_index = torch.stack([row,col]).to(self.device)
        for i in range(nattach):
            tmp_edge_index = erdos_renyi_graph(self.args.trigger_size,edge_prob=self.args.trigger_prob).to(self.device)
            tmp_edge_index[0] = start + i*self.args.trigger_size + tmp_edge_index[0]
            tmp_edge_index[1] = start + i*self.args.trigger_size + tmp_edge_index[1]

            trigger_row1_index = torch.cat([trojan_edge_index[0],tmp_edge_index[0]])
            trigger_row2_index = torch.cat([trojan_edge_index[1],tmp_edge_index[1]])
            trojan_edge_index = torch.cat([trigger_row1_index.unsqueeze(0),trigger_row2_index.unsqueeze(0)],0)
        trojan_edge_weights = torch.ones([trojan_edge_index.shape[1]],device=self.device,dtype=torch.float)
        
        '''
        construct features: randomly sample N nodes in targe class and assign their features to the trigget nodes
        '''
        injTriggerNum = nattach*self.args.trigger_size

        target_class_nodes_index = np.where(origin_labels.cpu() == self.args.target_class)[0]
        candidaite_nodes = [val for val in target_class_nodes_index if val in seen_node_idx]

        rs = np.random.RandomState(self.args.seed)
        trojan_feat_node_index = rs.choice(candidaite_nodes, size = injTriggerNum, replace=True)
        trojan_feat = x[trojan_feat_node_index].to(self.device)
        
        return trojan_feat, trojan_edge_index, trojan_edge_weights
# %%
