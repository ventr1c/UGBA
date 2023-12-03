from random import random
import torch 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from models.construct import model_construct


def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def obtain_attach_nodes(args,node_idxs, size):
    ### current random to implement
    size = min(len(node_idxs),size)
    rs = np.random.RandomState(args.seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice)
    return node_idxs[choice[:size]]

def obtain_attach_nodes_by_cluster(args,y_pred,model,node_idxs,x,labels,device,size):
    dis_weight = args.dis_weight
    cluster_centers = model.cluster_centers_
    distances = [] 
    distances_tar = []
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_tar_label = args.target_class
        
        tmp_center_x = cluster_centers[tmp_center_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[id].cpu().numpy())
        distances.append(dis)
        distances_tar.append(dis_tar)
        
    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
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

from torch_geometric.utils import degree
def obtain_attach_nodes_by_cluster_degree_all(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
    distances = [] 
    for id in range(x.shape[0]):
        tmp_center_label = y_pred[id]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    print(y_pred)
    nontarget_nodes = np.where(y_pred!=args.target_class)[0]

    non_target_node_idxs = np.array(list(set(nontarget_nodes) & set(node_idxs)))
    node_idxs = np.array(non_target_node_idxs)
    candiadate_distances = distances[node_idxs]
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    return selected_nodes
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

        single_labels_nodes_degrees = degrees[single_labels_nodes]
        single_labels_nodes_degrees = max_norm(single_labels_nodes_degrees)
        
        # the closer to the center, the more far away from the target centers
        # single_labels_dis_score =  single_labels_nodes_dis + dis_weight * (-single_labels_nodes_dis_tar)
        single_labels_dis_score = single_labels_nodes_dis + dis_weight * single_labels_nodes_degrees
        single_labels_nid_index = np.argsort(single_labels_dis_score) # sort descently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])
        if(label != label_list[-1]):
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes,sorted_single_labels_nodes[:last_seleced_num]])
    return candidate_nodes

import os
import time
from sklearn_extra import cluster
from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans, kmeans_predict

def cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    encoder_modelpath = './modelpath/{}_{}_benign.pth'.format('GCN_Encoder', args.dataset)
    if(os.path.exists(encoder_modelpath)):
        # load existing benign model
        gcn_encoder = torch.load(encoder_modelpath)
        gcn_encoder = gcn_encoder.to(device)
        edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Loading {} encoder Finished!".format(args.model))
    else:
        gcn_encoder = model_construct(args,'GCN_Encoder',data,device).to(device) 
        t_total = time.time()
        # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
        print("Length of training set: {}".format(len(idx_train)))
        gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
        print("Training encoder Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # # Save trained model
        # torch.save(gcn_encoder, encoder_modelpath)
        # print("Encoder saved at {}".format(encoder_modelpath))
    # test gcn encoder 
    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()
    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    gcn_encoder = gcn_encoder.cpu()
    kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
    kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    idx_attach = obtain_attach_nodes_by_cluster(args,y_pred,kmedoids,unlabeled_idx.cpu().tolist(),encoder_x,data.y,device,size).astype(int)
    return idx_attach

def cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    if(os.path.exists(selected_nodes_path)):
        print(selected_nodes_path)
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
        idx_attach = idx_attach[:size]
        return idx_attach
    gcn_encoder = model_construct(args,'GCN_Encoder',data,device).to(device) 
    t_total = time.time()
    # edge_weights = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()
    if(args.dataset == 'Cora' or args.dataset == 'Citeseer'):
        kmedoids = cluster.KMedoids(n_clusters=nclass,method='pam')
        kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=nclass,random_state=1)
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())

    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    idx_attach = obtain_attach_nodes_by_cluster_degree_all(args,train_edge_index,y_pred,cluster_centers,unlabeled_idx.cpu().tolist(),encoder_x,size).astype(int)
    selected_nodes_foldpath = "./selected_nodes/{}/Overall/seed{}".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_foldpath)):
        os.makedirs(selected_nodes_foldpath)
    selected_nodes_path = "./selected_nodes/{}/Overall/seed{}/nodes.txt".format(args.dataset,args.seed)
    if(not os.path.exists(selected_nodes_path)):
        np.savetxt(selected_nodes_path,idx_attach)
    else:
        idx_attach = np.loadtxt(selected_nodes_path, delimiter=',').astype(int)
    idx_attach = idx_attach[:size]
    return idx_attach

def obtain_attach_nodes_by_cluster_degree_single(args,edge_index,y_pred,cluster_centers,node_idxs,x,size):
    dis_weight = args.dis_weight
    degrees = (degree(edge_index[0])  + degree(edge_index[1])).cpu().numpy()
    distances = [] 
  
    for i in range(node_idxs.shape[0]):
        id = node_idxs[i]
        tmp_center_label = y_pred[i]
        tmp_center_x = cluster_centers[tmp_center_label]
        dis = np.linalg.norm(tmp_center_x - x[id].detach().cpu().numpy())
        distances.append(dis)
    distances = np.array(distances)
    print("y_pred",y_pred)
    print("node_idxs",node_idxs)

    candiadate_distances = distances
    candiadate_degrees = degrees[node_idxs]
    candiadate_distances = max_norm(candiadate_distances)
    candiadate_degrees = max_norm(candiadate_degrees)

    dis_score = candiadate_distances + dis_weight * candiadate_degrees
    candidate_nid_index = np.argsort(dis_score)
    sorted_node_idex = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idex
    print("selected_nodes",sorted_node_idex,selected_nodes)
    return selected_nodes

def cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device):
    gcn_encoder = model_construct(args,'GCN_Encoder',data,device).to(device) 
    t_total = time.time()
    print("Length of training set: {}".format(len(idx_train)))
    gcn_encoder.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    encoder_clean_test_ca = gcn_encoder.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Encoder CA on clean test nodes: {:.4f}".format(encoder_clean_test_ca))
    # from sklearn import cluster
    seen_node_idx = torch.concat([idx_train,unlabeled_idx])
    nclass = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x, train_edge_index,None).clone().detach()

    encoder_output = gcn_encoder(data.x,train_edge_index,None)
    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    cluster_centers = []
    each_class_size = int(size/(nclass-1))
    idx_attach = np.array([])
    for label in range(nclass):
        if(label == args.target_class):
            continue

        if(label != nclass-1):
            sing_class_size = each_class_size
        else:
            last_class_size= size - len(idx_attach)
            sing_class_size = last_class_size
        idx_sing_class = (y_pred == label).nonzero()[0]
        print("idx_sing_class",idx_sing_class)
        if(len(idx_sing_class) == 0):
            continue
        print("current_class_size",sing_class_size)
        selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset,args.seed,label)
        if(os.path.exists(selected_nodes_path)):
            print(selected_nodes_path)
            sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
            print(sing_idx_attach)
            sing_idx_attach = sing_idx_attach[:sing_class_size]
            idx_attach = np.concatenate((idx_attach,sing_idx_attach))
        else:
            kmedoids = KMeans(n_clusters=2,random_state=1)
            kmedoids.fit(encoder_x[idx_sing_class].detach().cpu().numpy())
            sing_center = kmedoids.cluster_centers_
            cluster_ids_x = kmedoids.predict(encoder_x[idx_sing_class].cpu().numpy())
            cand_idx_sing_class = np.array(list(set(unlabeled_idx.cpu().tolist())&set(idx_sing_class)))
            if(label != nclass - 1):
                sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,each_class_size).astype(int)
                selected_nodes_foldpath = "./selected_nodes/{}/Seperate/seed{}".format(args.dataset,args.seed)
                if(not os.path.exists(selected_nodes_foldpath)):
                    os.makedirs(selected_nodes_foldpath)
                selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset,args.seed,label)
                if(not os.path.exists(selected_nodes_path)):
                    np.savetxt(selected_nodes_path,sing_idx_attach)
                else:
                    sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
                sing_idx_attach = sing_idx_attach[:each_class_size]
            else:
                last_class_size= size - len(idx_attach)
                sing_idx_attach = obtain_attach_nodes_by_cluster_degree_single(args,train_edge_index,cluster_ids_x,sing_center,cand_idx_sing_class,encoder_x,last_class_size).astype(int)
                selected_nodes_path = "./selected_nodes/{}/Seperate/seed{}/class_{}.txt".format(args.dataset,args.seed,label)
                np.savetxt(selected_nodes_path,sing_idx_attach)
                if(not os.path.exists(selected_nodes_path)):
                    np.savetxt(selected_nodes_path,sing_idx_attach)
                else:
                    sing_idx_attach = np.loadtxt(selected_nodes_path, delimiter=',')
                sing_idx_attach = sing_idx_attach[:each_class_size]
            idx_attach = np.concatenate((idx_attach,sing_idx_attach))





    return idx_attach
