# %%
from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
from models.GCN_Encoder import GCN_Encoder

from GNNGuard.GCN import GuardGCN
from MedianGCN.GCN import MedianGCN
# from deeprobust.graph.defense import MedianGCN

def model_construct(args,model_name,data,device):
    if (model_name == 'GCN'):
        if(args.dataset == 'Reddit2'):
            use_ln = True
            layer_norm_first = True
        else:
            use_ln = False
            layer_norm_first = False

        model = GCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device,
                    use_ln=use_ln,
                    layer_norm_first=layer_norm_first)
        
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
        if(args.dataset == 'Reddit2'):
            use_ln = True
            layer_norm_first = False
        else:
            use_ln = False
            layer_norm_first = False
        model = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=args.hidden,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=args.dropout,                    
                            lr=args.train_lr,                    
                            weight_decay=args.weight_decay,                    
                            device=device,
                            use_ln=use_ln,
                            layer_norm_first=layer_norm_first)
    return model

def defend_baseline_construct(args,defend_method,model_name,data,device):
    if (model_name == 'GCN'):
        if(args.dataset == 'Reddit2'):
            use_ln = True
            layer_norm_first = False
        else:
            use_ln = False
            layer_norm_first = False
        if(defend_method == 'guard'):
            model = GuardGCN(nfeat=data.x.shape[1],\
                        nhid=args.hidden,\
                        nclass= int(data.y.max()+1),\
                        dropout=args.dropout,\
                        lr=args.train_lr,\
                        weight_decay=args.weight_decay,\
                        device=device,
                        use_ln=use_ln,
                        layer_norm_first=layer_norm_first)
        elif(defend_method == 'median'):
            model = MedianGCN(nfeat=data.x.shape[1],\
                        nhid=args.hidden,\
                        nclass= int(data.y.max()+1),\
                        dropout=args.dropout,\
                        lr=args.train_lr,\
                        weight_decay=args.weight_decay,\
                        device=device)
        else:
            model = GCN(nfeat=data.x.shape[1],\
                        nhid=args.hidden,\
                        nclass= int(data.y.max()+1),\
                        dropout=args.dropout,\
                        lr=args.train_lr,\
                        weight_decay=args.weight_decay,\
                        device=device,
                        use_ln=use_ln,
                        layer_norm_first=layer_norm_first)
        return model
        
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
        if(args.dataset == 'Reddit2'):
            use_ln = True
            layer_norm_first = False
        else:
            use_ln = False
            layer_norm_first = False
        model = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=args.hidden,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=args.dropout,                    
                            lr=args.train_lr,                    
                            weight_decay=args.weight_decay,                    
                            device=device,
                            use_ln=use_ln,
                            layer_norm_first=layer_norm_first)
    return model