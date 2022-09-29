
# Attack: Clean Graph
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'none' --device_id 0 > ./logs/Pubmed/Pubmed_Clean_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'prune' --prune_thr 0.2 --device_id 0 > ./logs/Pubmed/Pubmed_Clean_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'isolate' --prune_thr 0.2 --device_id 0 > ./logs/Pubmed/Pubmed_Clean_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Samp.
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'none' --evaluate_mode 'overall' --device_id 0 > ./logs/Pubmed/Pubmed_Rand_Samp_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'prune' --prune_thr 0.2 --evaluate_mode 'overall' --device_id 0 > ./logs/Pubmed/Pubmed_Rand_Samp_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'isolate' --prune_thr 0.2 --evaluate_mode 'overall' --device_id 0 > ./logs/Pubmed/Pubmed_Rand_Samp_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Gene.
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'none' --evaluate_mode 'overall' --device_id 0 > ./logs/Pubmed/Pubmed_Rand_Gene_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'prune' --prune_thr 0.2 --evaluate_mode 'overall' --device_id 0 > ./logs/Pubmed/Pubmed_Rand_Gene_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'isolate' --prune_thr 0.2 --evaluate_mode 'overall' --device_id 0 > ./logs/Pubmed/Pubmed_Rand_Gene_Isolate_gcn2gcn.log 2>&1 &

# Attack: GTA
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --device_id 1 > ./logs/Pubmed/Pubmed_Basic_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --device_id 1 > ./logs/Pubmed/Pubmed_Basic_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.2 --device_id 1 > ./logs/Pubmed/Pubmed_Basic_Isolate_gcn2gcn.log 2>&1 &

# Attack: Ours
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Ours_Isolate_gcn2gcn.log 2>&1 &

# Attack: without selection
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Pubmed/Pubmed_Unno_Isolate_gcn2gcn.log 2>&1 &

# Attack: without constraints 
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Sel_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Sel_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.2 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Sel_Isolate_gcn2gcn.log 2>&1 &

# Attack: Varient Attach node: Ours 
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 5 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.5.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 10 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.10.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 15 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.15.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 20 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.20.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'none' --trojan_epochs 2000 --homo_loss_weight 0 --homo_boost_thrd 0.5 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.40.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 60 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.60.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.80.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 100 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.100.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 120 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_None_gcn2gcn_clu_vs.120.log 2>&1 &

## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 5 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs5.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 10 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs10.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 15 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs15.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 20 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs20.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'prune' --trojan_epochs 2000 --prune_thr 0.2 --homo_loss_weight 50 --homo_boost_thrd 0.5 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs40_1.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 60 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs60.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs80.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 100 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs100.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 120 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 100 --dis_weight 0 --selection_method 'cluster_degree' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Prune_gcn2gcn_clu_vs120.log 2>&1 &

## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'isolate' --trojan_epochs 2000 --prune_thr 0.2 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 0 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Ours_Isolate_gcn2gcn_clu_vs40.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Pubmed/Pubmed_Ours_Isolate_gcn2gcn.log 2>&1 &

# Attack: Varient Attach node: Without selection
## Defend: None
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 5 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs5.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 10 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs10.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 15 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs15.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 20 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs20.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'none' --trojan_epochs 400 --prune_thr 0.2 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 1 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs40.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 60 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs60.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs80.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 100 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs100.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 120 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 100 --dis_weight 1 --selection_method 'none' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_None_gcn2gcn_vs120.log 2>&1 &
## Defend: Prune
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 5 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs5.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 10 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs10.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 15 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs15.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 20 --attack_method 'Basic' --defense_mode 'prune' --trojan_epochs 400 --prune_thr 0.2 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs20.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'prune' --trojan_epochs 400 --prune_thr 0.2 --homo_loss_weight 100 --homo_boost_thrd 0.5 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 1 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs40.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 60 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 2 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs60.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs80.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 100 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs100.log 2>&1 &
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --seed 10 --use_vs_number --vs_number 120 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --trojan_epochs 400 --selection_method 'none' --device_id 3 --evaluate_mode '1by1' > ./logs/Pubmed/Pubmed_Unno_Prune_gcn2gcn_vs120.log 2>&1 &

## Defend: Isolated
nohup python -u run_adaptive.py --dataset 'Pubmed' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.2 --homo_loss_weight 3 --dis_weight 1 --selection_method 'none' --device_id 2 > ./logs/Pubmed/Pubmed_Unno_Isolate_gcn2gcn.log 2>&1 &
