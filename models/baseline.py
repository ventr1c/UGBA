#%%
import torch

#%%
import numpy as np
from torch_geometric.utils import erdos_renyi_graph
class BaseLine:

    def __init__(self,args, device, features):
        self.args = args
        self.device = device
        self.trigger_size = args.trigger_size
        self.trojan_feat, self.trojan_edge_index = self.gene_trigger(features)
        

    def gene_trigger(self,features):
        trojan_edge_index = erdos_renyi_graph(self.trigger_size,edge_prob=self.args.trigger_prob).to(self.device)
        
        # trojan_feat = torch.zeros([self.trigger_size,features.shape[1]]).to(self.device)
        rs = np.random.RandomState(self.args.seed)
        if self.args.attack_method == 'Rand_Gene':
            print("Rand generate the trigger")
            features = features.cpu().numpy()
            mean = features.mean(axis=0)
            std = features.std(axis=0)

            trojan_feat = []
            for i in range(self.trigger_size):
                trojan_feat.append(torch.tensor(rs.normal(mean,std),dtype=torch.float32,device=self.device))
            trojan_feat = torch.stack(trojan_feat)
        else:
            print("Rand sample the trigger")
            idx = rs.randint(features.shape[0],size=self.trigger_size)
            trojan_feat = features[idx]

        return trojan_feat, trojan_edge_index
    
    def get_poisoned_rand(self,features,edge_index,labels,idx_attach):

        trojan_labels = labels.clone()
        trojan_labels[idx_attach] = self.args.target_class
        trojan_features,trojan_edge_index,weights = self.inject_trigger_rand(idx_attach,features,edge_index)


        return trojan_features,trojan_edge_index,weights,trojan_labels

    def inject_trigger_rand(self, idx_attach, features, edge_index):

        edge_list = []
        start = features.shape[0]
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*self.args.trigger_size])
        trojan_edge_index = torch.tensor(edge_list,device=self.device,dtype=torch.long).T
        for i in range(len(idx_attach)):
            tmp_edge_index = self.trojan_edge_index.clone()
            tmp_edge_index[0] = start + i*self.args.trigger_size + tmp_edge_index[0]
            tmp_edge_index[1] = start + i*self.args.trigger_size + tmp_edge_index[1]
            trojan_edge_index = torch.cat([trojan_edge_index,tmp_edge_index],dim=1)
        
        
        from torch_geometric.utils import to_undirected
        trojan_edge_index = to_undirected(trojan_edge_index)
        trojan_edge_index = torch.cat([edge_index,trojan_edge_index],dim=1)

        trojan_features = self.trojan_feat.repeat([len(idx_attach),1])
        trojan_features = torch.cat([features,trojan_features])
        weights= torch.ones([trojan_edge_index.shape[1]],dtype=torch.float32,device=self.device)

        return trojan_features,trojan_edge_index,weights
        
