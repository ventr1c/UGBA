# Attack: Clean Graph
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'none' --device_id 0 > ./logs/Arxiv/Arxiv_Clean_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'prune' --prune_thr 0.3 --device_id 0 > ./logs/Arxiv/Arxiv_Clean_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'isolate' --prune_thr 0.3 --device_id 3 > ./logs/Arxiv/Arxiv_Clean_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Samp.
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'none' --evaluate_mode 'overall' --device_id 2 > ./logs/Arxiv/Arxiv_Rand_Samp_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'prune' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 2 > ./logs/Arxiv/Arxiv_Rand_Samp_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'isolate' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 2 > ./logs/Arxiv/Arxiv_Rand_Samp_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Gene.
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'none' --evaluate_mode 'overall' --device_id 0 > ./logs/Arxiv/Arxiv_Rand_Gene_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'prune' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 0 > ./logs/Arxiv/Arxiv_Rand_Gene_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'isolate' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 3 > ./logs/Arxiv/Arxiv_Rand_Gene_Isolate_gcn2gcn.log 2>&1 &

# Attack: GTA
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --device_id 0 > ./logs/Arxiv/Arxiv_Basic_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --device_id 3 > ./logs/Arxiv/Arxiv_Basic_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.3 --device_id 0 > ./logs/Arxiv/Arxiv_Basic_Isolate_gcn2gcn.log 2>&1 &

# Attack: Ours
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Ours_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Ours_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.3 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Ours_Isolate_gcn2gcn.log 2>&1 &

# Attack: Without Selection
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Arxiv/Arxiv_Unno_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.3 --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Arxiv/Arxiv_Unno_Isolate_gcn2gcn.log 2>&1 &

# Attack: Without Constraint
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Sel_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Sel_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.3 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Sel_Isolate_gcn2gcn.log 2>&1 &

# Attack: Variance Ours
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --use_vs_number --vs_number 80 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Arxiv/Arxiv_Ours_None_gcn2gcn_80.log 2>&1 &

# Attack: Without Selection
## Defend: None
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 0 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs40.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 1 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs80.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 120 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs120.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 160 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 3 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs160.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 200 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 1 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs200.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 240 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs240.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 320 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 3 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs320.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 400 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 0 > ./logs/Arxiv/Arxiv_Unno_None_gcn2gcn_vs400.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 0 > ./logs/Arxiv/Arxiv_Unno_Prune_gcn2gcn_vs80.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 160 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 1 > ./logs/Arxiv/Arxiv_Unno_Prune_gcn2gcn_vs160.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 240 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Arxiv/Arxiv_Unno_Prune_gcn2gcn_vs240.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 320 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 3 > ./logs/Arxiv/Arxiv_Unno_Prune_gcn2gcn_vs320.log 2>&1 &
nohup python run_adaptive.py --dataset 'ogbn-arxiv' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 400 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 0 > ./logs/Arxiv/Arxiv_Unno_Prune_gcn2gcn_vs400.log 2>&1 &