from Baseline_Attack.attacks.seqgia import SEQGIA

def baseline_attack_parser(args,device):
    if(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
        atk_param = {'attack_lr': 0.01,
                    'attack_epoch': 500,
                    'agia_epoch': 300,
                    'n_inject_max': None,
                    'n_edge_max': None,
                    'feat_lim_min': -1,
                    'feat_lim_max': 1,
                    'device': device,
                    'early_stop': 200,
                    'disguise_coe': 1.0,
                    'sequential_step': 0.2,
                    'injection': 'tdgia',
                    'feat_upd': 'gia',
                    'branching': False,
                    'iter_epoch': 2,
                    'agia_pre': 0.5, 
                    'hinge': True
                    }
        return atk_param

def attack_baseline_construct(args,parm):
    if(args.attack_method == 'TDGIA' or args.attack_method == 'AGIA'):
        attacker = SEQGIA(epsilon=parm['attack_lr'],
                    n_epoch=parm['attack_epoch'],
                    a_epoch=parm['agia_epoch'],
                    n_inject_max= parm['n_inject_max'],
                    n_edge_max= parm['n_edge_max'],
                    feat_lim_min=parm['feat_lim_min'],
                    feat_lim_max=parm['feat_lim_max'],
                    device=parm['device'],
                    early_stop=parm['early_stop'],
                    disguise_coe=parm['disguise_coe'],
                    sequential_step=parm['sequential_step'],
                    injection=parm['injection'],
                    feat_upd=parm['feat_upd'],
                    branching=parm['branching'],
                    iter_epoch=parm['iter_epoch'],
                    agia_pre=parm['agia_pre'],
                    hinge=parm['hinge'])
        return attacker