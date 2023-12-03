# %%
from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage
from models.GCN_Encoder import GCN_Encoder
from models.GNNGuard import GNNGuard
from models.RobustGCN import RobustGCN

def model_construct(args,model_name,data,device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    if (model_name == 'GCN'):

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
        model = GCN_Encoder(nfeat=data.x.shape[1],                    
                            nhid=args.hidden,                    
                            nclass= int(data.y.max()+1),                    
                            dropout=args.dropout,                    
                            lr=args.train_lr,                    
                            weight_decay=args.weight_decay,                    
                            device=device,
                            use_ln=use_ln,
                            layer_norm_first=layer_norm_first)
    elif(model_name == 'GNNGuard'):
        model = GNNGuard(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    use_ln=use_ln,\
                    device=device)
    elif(model_name == 'RobustGCN'):
        model = RobustGCN(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    device=device)
    else:
        print("Not implement {}".format(model_name))
    return model
