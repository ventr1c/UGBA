#%%
import torch

#%%
import numpy as np
from torch_geometric.utils import erdos_renyi_graph
class BaseLine:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.poison_x = None
        self.poison_edge_index = None
        self.poison_edge_weights = None

    def inject_trigger_rand(self, idx_attach, features,edge_index, edge_weight, labels, full_data = False):

        if(self.args.attack_method == 'Rand_Gene'):
            trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_gene(features, self.idx_labeled, idx_attach, full_data)
        elif(self.args.attack_method == 'Rand_Samp'):
            trojan_feat, trojan_edge_index, trojan_weights = self.trigger_rand_samp(features, self.idx_labeled, labels, idx_attach, full_data)
        update_edge_index = torch.cat([edge_index,trojan_edge_index],dim=1)
        update_edge_weights = torch.cat([edge_weight,trojan_weights]) 
        update_feat = torch.cat([features,trojan_feat])

        return update_feat, update_edge_index, update_edge_weights
    
    def get_poisoned_rand(self,features,edge_index,edge_weights,labels,idx_train,idx_attach,unlabeled_idx):
        self.features = features
        self.edge_index = edge_index
        if edge_weights is None:
            edge_weights = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.edge_weights = edge_weights
        self.labels = labels
        self.idx_train = idx_train
        self.idx_attach = idx_attach
        self.unlabeled_idx = unlabeled_idx
        self.idx_labeled = torch.concat([idx_train,unlabeled_idx]).to(self.device)
        # change idx_attach to target label
        self.origin_labels = labels.clone()
        self.labels[idx_attach] = self.args.target_class

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger_rand(self.idx_attach,self.features,self.edge_index,self.edge_weights, self.origin_labels)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

    def trigger_rand_gene(self, x, idx_labeled, idx_attach, full_data = True):

        nattach = len(idx_attach)
        start = x.shape[0]

        edge_list = []
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*self.args.trigger_size])

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
        # calculate the average of node features
        if(full_data == True):
            nodeNum = x.shape[0]
            featDim = x.shape[1]
            aver_featNum = int(len(x.nonzero())/nodeNum)
        else:
            nodeNum = x[idx_labeled].shape[0]
            featDim = x[idx_labeled].shape[1]
            aver_featNum = int(len(x[idx_labeled].nonzero())/nodeNum)
        nattach = len(idx_attach)
        # construct features: randomly sample N dims in one feature 
        # vectors and assign their items as 1
    
        injTriggerNum = nattach*self.args.trigger_size
        trojan_feat = torch.zeros((injTriggerNum,featDim)).to(self.device)
        rs = np.random.RandomState(self.args.seed)
        # seeds = list(rs.choice(10000,size = injTriggerNum))
        for i in range(injTriggerNum):
            # rs = np.random.RandomState(seeds[i])
            trojan_feat_dim_index = rs.choice(featDim, size = aver_featNum, replace=False)
            trojan_feat[i][trojan_feat_dim_index] = 1 
        
        return trojan_feat, trojan_edge_index, trojan_edge_weights

    def trigger_rand_samp(self, x, seen_node_idx, origin_labels, idx_attach, full_data = True):
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
        
        # construct features: randomly sample N nodes in targe class and 
        # assign their features to the trigget nodes

        injTriggerNum = nattach*self.args.trigger_size

        target_class_nodes_index = np.where(origin_labels.cpu() == self.args.target_class)[0]
        candidaite_nodes = [val for val in target_class_nodes_index if val in seen_node_idx]

        rs = np.random.RandomState(self.args.seed)
        trojan_feat_node_index = rs.choice(candidaite_nodes, size = injTriggerNum, replace=True)
        trojan_feat = x[trojan_feat_node_index].to(self.device)
        
        return trojan_feat, trojan_edge_index, trojan_edge_weights


