# Attack: Clean Graph
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'none' --device_id 1 > ./logs/Flickr_poi001_Clean_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'prune' --prune_thr 0.3 --device_id 1 > ./logs/Flickr_poi001_Clean_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'None' --defense_mode 'isolate' --prune_thr 0.3 --device_id 1 > ./logs/Flickr_poi001_Clean_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Samp.
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'none' --evaluate_mode 'overall' --device_id 2 > ./logs/Flickr_poi001_Rand_Samp_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'prune' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 2 > ./logs/Flickr_poi001_Rand_Samp_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Samp' --defense_mode 'isolate' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 2 > ./logs/Flickr_poi001_Rand_Samp_Isolate_gcn2gcn.log 2>&1 &

# Attack: Rand.Gene.
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'none' --evaluate_mode 'overall' --device_id 1 > ./logs/Flickr_poi001_Rand_Gene_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'prune' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 1 > ./logs/Flickr_poi001_Rand_Gene_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Rand_Gene' --defense_mode 'isolate' --prune_thr 0.3 --evaluate_mode 'overall' --device_id 1 > ./logs/Flickr_poi001_Rand_Gene_Isolate_gcn2gcn.log 2>&1 &

# Attack: GTA
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --device_id 1 > ./logs/Flickr_poi001_Basic_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --device_id 1 > ./logs/Flickr_poi001_Basic_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.3 --device_id 1 > ./logs/Flickr_poi001_Basic_Isolate_gcn2gcn.log 2>&1 &

# Attack: Ours
## Defend: None
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Flickr_poi001_Ours_None_gcn2gcn.log 2>&1 &
## Defend: Prune
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --prune_thr 0.3 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Flickr_poi001_Ours_Prune_gcn2gcn.log 2>&1 &
## Defend: Isolated
nohup python run_adaptive.py --dataset 'Flickr' --train_lr 0.02 --hidden 128 --vs_ratio 0.001 --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --prune_thr 0.3 --homo_loss_weight 3 --dis_weight 1 --selection_method 'cluster_degree' --device_id 2 > ./logs/Flickr_poi001_Ours_Isolate_gcn2gcn.log 2>&1 &

