# Attack: Clean Graph
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'none' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr_Clean_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'prune' --prune_thr 0.4 --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr_Clean_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'isolate' --prune_thr 0.4 --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr_Clean_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Samp.
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'none' --evaluate_mode '1by1' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr_Rand_Samp_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'prune' --prune_thr 0.4 --evaluate_mode '1by1' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr_Rand_Samp_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'isolate' --prune_thr 0.4 --evaluate_mode '1by1' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr_Rand_Samp_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Gene.
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'none' --evaluate_mode '1by1' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Rand_Gene_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'prune' --prune_thr 0.4 --evaluate_mode '1by1' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Rand_Gene_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'isolate' --prune_thr 0.4 --evaluate_mode '1by1' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Rand_Gene_Isolate_gcn2gcn.log 2>&1 &

# Attack: GTA
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Basic_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Basic_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.4 --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Basic_Isolate_gcn2gcn.log 2>&1 &

# Attack: Ours
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Ours_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 1 --dis_weight 1 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr_Ours_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.4 --homo_loss_weight 1 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Ours_Isolate_gcn2gcn.log 2>&1 &

# Attack: Without selection
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Unno_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Unno_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.4 --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Unno_Isolate_gcn2gcn.log 2>&1 &

# Attack: Without constraints
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Sel_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Sel_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.4 --homo_loss_weight 0 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr_Sel_Isolate_gcn2gcn.log 2>&1 &

# Attack: Varient Attach node: Ours
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'none' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'cluster_degree' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Ours_None_gcn2gcn_clu_vs40.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'none' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 1000 --selection_method 'cluster_degree' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Ours_None_gcn2gcn_clu_vs80.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'cluster_degree' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Ours_Prune_gcn2gcn_clu_vs80.log 2>&1 &
## Defend: Isolate
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'cluster_degree' --device_id 3 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Ours_Isolate_gcn2gcn_clu_vs80.log 2>&1 &

# Attack: Varient Attach node: Without Selection
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_None_gcn2gcn_clu_vs5.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 160 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_None_gcn2gcn_vs10.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 240 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 3 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_None_gcn2gcn_vs20.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 320 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_None_gcn2gcn_vs40.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 400 --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --dis_weight 1 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_None_gcn2gcn_vs80.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 40 --attack_method 'Basic' --defense_mode 'prune'  --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'none' --device_id 1 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_Prune_gcn2gcn_vs40.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 80 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_Prune_gcn2gcn_vs80.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 120 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'none' --device_id 3 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_Prune_gcn2gcn_vs120.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 160 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'none' --device_id 0 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_Prune_gcn2gcn_vs160.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 64 --model 'GCN' --test_model 'GCN' --use_vs_number --vs_number 200 --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.4 --homo_loss_weight 100 --dis_weight 0 --trojan_epochs 400 --selection_method 'none' --device_id 2 --evaluate_mode "1by1" > ./logs/Flickr/Flickr_Unno_Prune_gcn2gcn_vs200.log 2>&1 &