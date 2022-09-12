#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
def model_construct(args,model_name,data,device):
    if (model_name == 'GCN'):
        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.lr,\
                    weight_decay=args.weight_decay,\
                    device=device)
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(data.y.max()+1), 
                    heads=8,
                    dropout=args.dropout, 
                    lr=args.lr, 
                    weight_decay=args.weight_decay, 
                    device=device)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.lr,\
                weight_decay=args.weight_decay,\
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
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.05):
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

    def forward(self, input, thrd, binaryfeat=False):

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
        feat = GW(feat, thrd, self.device)
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
        return loss

#%%
import numpy as np
import torch.optim as optim
from models.GCN import GCN
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

from torch_geometric.utils import to_undirected
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
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight):

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        return update_feat, update_edge_index, update_edge_weights


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)

        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=self.args.dropout, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
        # change the labels of the poisoned node to the target class
        labels = labels.clone()
        labels[idx_attach] = args.target_class
        self.labels = labels

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        self.poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        self.trojan.train()
        for i in range(args.trojan_epochs):

            optimizer_shadow.zero_grad()
            optimizer_trigger.zero_grad()

            trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
            trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)

            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,features.shape[1]])
            self.poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
            self.poison_x = torch.cat([features,trojan_feat])

            output = self.shadow_model(self.poison_x, self.poison_edge_index, self.poison_edge_weights)
            
            loss_train = F.nll_loss(output[torch.cat([idx_train,idx_attach])], labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
            
            loss_homo = 0.0
            if(self.args.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                            trojan_weights,\
                                            self.poison_x,\
                                            self.args.homo_boost_thrd)
            
            loss_all = loss_train + self.args.homo_loss_weight * loss_homo

            loss_all.backward()

            optimizer_shadow.step()
            optimizer_trigger.step()

            acc_train_clean = utils.accuracy(output[idx_train], labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], labels[idx_attach])
            

            if args.debug and i % 10 == 0:
                print('Epoch {}, training loss: {:.5f}, homo loss: {:.5f} '.format(i, loss_train, loss_homo))
                print("acc_train_clean: {:.4f}, acc_train_attach: {:.4f}".format(acc_train_clean,acc_train_attach))
        self.trojan.eval()
    
    def get_poisoned(self):

        poison_x = self.poison_x.data
        poison_edge_index = self.poison_edge_index.data
        poison_edge_weights = self.poison_edge_weights.data
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        poison_labels = self.labels

        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

