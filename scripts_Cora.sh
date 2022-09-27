
# Attack: Clean Graph
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'none' --device_id 0 > ./logs/Cora_Clean_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'prune' --prune_thr 0.1 --device_id 0 > ./logs/Cora_Clean_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'isolate' --prune_thr 0.1 --device_id 0 > ./logs/Cora_Clean_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Samp.
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'none' --evaluate_mode 'overall' --device_id 0 > ./logs/Cora_Rand_Samp_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'prune' --prune_thr 0.1 --evaluate_mode 'overall' --device_id 0 > ./logs/Cora_Rand_Samp_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'isolate' --prune_thr 0.1 --evaluate_mode 'overall' --device_id 0 > ./logs/Cora_Rand_Samp_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Gene.
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'none' --evaluate_mode 'overall' --device_id 0 > ./logs/Cora_Rand_Gene_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'prune' --prune_thr 0.1 --evaluate_mode 'overall' --device_id 0 > ./logs/Cora_Rand_Gene_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'isolate' --prune_thr 0.1 --evaluate_mode 'overall' --device_id 0 > ./logs/Cora_Rand_Gene_Isolate_gcn2gcn.log 2>&1 &

# Attack: GTA
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --device_id 0 > ./logs/Cora_Basic_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.1 --device_id 0 > ./logs/Cora_Basic_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.1 --device_id 0 > ./logs/Cora_Basic_Isolate_gcn2gcn.log 2>&1 &

# Attack: Ours
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 2 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Ours_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.1 --homo_loss_weight 2 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Ours_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.1 --homo_loss_weight 2 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Ours_Isolate_gcn2gcn.log 2>&1 &

# Attack: without selection
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 2 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Cora_Unno_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.1 --homo_loss_weight 2 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Cora_Unno_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.1 --homo_loss_weight 2 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Cora_Unno_Isolate_gcn2gcn.log 2>&1 &

# Attack: without constraints 
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Sel_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.1 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Sel_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.1 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Sel_Isolate_gcn2gcn.log 2>&1 &

# Attack: Ours
## Defend: None
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 2 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Ours_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.1 --homo_loss_weight 2 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Ours_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.1 --homo_loss_weight 2 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Cora_Ours_Isolate_gcn2gcn.log 2>&1 &

# Attack: Variance Ours
# Defend: None
nohup python -u run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 3 --attack_method 'Basic' --defense_mode 'none' --trojan_epochs 2000 --prune_thr 0.1 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Cora/Cora_Ours_None_gcn2gcn_clu_vs2.log 2>&1 &
# Defend: Prune
nohup python -u run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 3 --attack_method 'Basic' --defense_mode 'prune' --trojan_epochs 2000 --prune_thr 0.1 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Cora/Cora_Ours_Prune_gcn2gcn_clu_vs2.log 2>&1 &
# Defend: Isolate
nohup python -u run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 3 --attack_method 'Basic' --defense_mode 'isolate' --trojan_epochs 2000 --prune_thr 0.1 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Cora/Cora_Ours_Isolate_gcn2gcn_clu_vs2.log 2>&1 &