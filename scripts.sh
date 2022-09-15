'''basic'''
# basic backdoor attack\defense
nohup python run_basic.py --dataset 'cora' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 5 > ./logs/basic_cora_atk_gcn2gcn.log 2>&1 &
nohup python run_basic.py --dataset 'cora' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 > ./logs/basic_cora_pru_gcn2gcn.log 2>&1 &
nohup python run_basic.py --dataset 'cora' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 > ./logs/basic_cora_iso_gcn2gcn.log 2>&1 &

nohup python run_basic.py --dataset 'citeseer' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 5 > ./logs/basic_citeseer_atk_gcn2gcn.log 2>&1 &
nohup python run_basic.py --dataset 'citeseer' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 > ./logs/basic_citeseer_pru_gcn2gcn.log 2>&1 &
nohup python run_basic.py --dataset 'citeseer' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 > ./logs/basic_citeseer_iso_gcn2gcn.log 2>&1 &

nohup python run_basic.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 5 > ./logs/basic_pubmed_atk_gcn2gcn.log 2>&1 &
nohup python run_basic.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 > ./logs/basic_pubmed_pru_gcn2gcn.log 2>&1 &
nohup python run_basic.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 > ./logs/basic_pubmed_iso_gcn2gcn.log 2>&1 &

nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_pubmed_pru_gcn2gcn_pr015.log 2>&1 &
nohup python run_basic.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_pubmed_iso_gcn2gcn_pr015.log 2>&1 &
'''transfer'''
## unno transfer: to gat
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GAT' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_cora_atk_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GAT' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.1> ./logs/unno_cora_pru_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GAT' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.1 > ./logs/unno_cora_iso_gcn2gat.log 2>&1 &

nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GAT' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_citeseer_atk_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GAT' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_citeseer_pru_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GAT' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_citeseer_iso_gcn2gat.log 2>&1 &

nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GAT' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_pubmed_atk_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GAT' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15> ./logs/unno_pubmed_pru_gcn2gat_pr015.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GAT' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_pubmed_iso_gcn2gat_pr015.log 2>&1 &

## unno transfer: to sage
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GraphSage' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_cora_atk_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GraphSage' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.1> ./logs/unno_cora_pru_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GraphSage' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.1 > ./logs/unno_cora_iso_gcn2sage.log 2>&1 &

nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GraphSage' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_citeseer_atk_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GraphSage' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15> ./logs/unno_citeseer_pru_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GraphSage' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_citeseer_iso_gcn2sage.log 2>&1 &

nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GraphSage' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_pubmed_atk_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GraphSage' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15> ./logs/unno_pubmed_pru_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GraphSage' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_pubmed_iso_gcn2sage.log 2>&1 &

## basic transfer: to gat
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GAT' --defense_mode 'none' --device_id 5 --homo_loss_weight 0 > ./logs/basic_cora_atk_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GAT' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.1> ./logs/basic_cora_pru_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GAT' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.1 > ./logs/basic_cora_iso_gcn2gat.log 2>&1 &

nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GAT' --defense_mode 'none' --device_id 5 --homo_loss_weight 0 > ./logs/basic_citeseer_atk_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GAT' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_citeseer_pru_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GAT' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_citeseer_iso_gcn2gat.log 2>&1 &

nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GAT' --defense_mode 'none' --device_id 5 --homo_loss_weight 0 > ./logs/basic_pubmed_atk_gcn2gat.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GAT' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15> ./logs/basic_pubmed_pru_gcn2gat_pr015.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GAT' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_pubmed_iso_gcn2gat_pr015.log 2>&1 &

## basic transfer: to sage
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GraphSage' --defense_mode 'none' --device_id 5 --homo_loss_weight 0 > ./logs/basic_cora_atk_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GraphSage' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.1> ./logs/basic_cora_pru_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'cora' --model 'GCN' --test_model 'GraphSage' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.1 > ./logs/basic_cora_iso_gcn2sage.log 2>&1 &

nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GraphSage' --defense_mode 'none' --device_id 5 --homo_loss_weight 0 > ./logs/basic_citeseer_atk_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GraphSage' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15> ./logs/basic_citeseer_pru_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GraphSage' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_citeseer_iso_gcn2sage.log 2>&1 &

nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GraphSage' --defense_mode 'none' --device_id 5 --homo_loss_weight 0 > ./logs/basic_pubmed_atk_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GraphSage' --defense_mode 'prune' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15> ./logs/basic_pubmed_pru_gcn2sage.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GraphSage' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 0 --prune_thr 0.15 > ./logs/basic_pubmed_iso_gcn2sage.log 2>&1 &

'''only add unno constraint'''
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 --selection_method 'none' > ./logs/unno_Cora_atk_gcn2gcn.log 2>&1 &
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --selection_method 'none' > ./logs/unno_Cora_pru_gcn2gcn.log 2>&1 &
nohup python run_adaptive.py --dataset 'Cora' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --selection_method 'none' > ./logs/unno_Cora_iso_gcn2gcn.log 2>&1 &

nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_citeseer_atk_gcn2gcn.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_citeseer_pru_gcn2gcn_pr015.log 2>&1 &
nohup python run_adaptive.py --dataset 'citeseer' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_citeseer_iso_gcn2gcn_pr015.log 2>&1 &

nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 5 --homo_loss_weight 1 > ./logs/unno_pubmed_atk_gcn2gcn.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'prune' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_pubmed_pru_gcn2gcn_pr015.log 2>&1 &
nohup python run_adaptive.py --dataset 'pubmed' --model 'GCN' --test_model 'GCN' --defense_mode 'isolate' --device_id 5 --homo_loss_weight 1 --prune_thr 0.15 > ./logs/unno_pubmed_iso_gcn2gcn_pr015.log 2>&1 &

'''Rand Attack baseline'''
nohup python run_adaptive.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --defense_mode 'none' --device_id 1 --homo_loss_weight 1 --attack_method 'Rand_Gene' --selection_method 'none' > ./logs/rand_Flickr_atk_gcn2gcn.log 2>&1 &
'''full version of adaptive'''
nohup python run_adaptive.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'GTA' --defense_mode 'none' --device_id 1 --homo_loss_weight 0 --selection_method 'cluster' --prune_thr 0.15 > ./logs/ada_Flickr_atk_gcn2gcn.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'GTA' --defense_mode 'prune' --device_id 1 --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.15 > ./logs/ada_Flickr_pru_gcn2gcn.log 2>&1 &
nohup python run_adaptive.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'GTA' --defense_mode 'isolate' --device_id 1 --homo_loss_weight 1 --selection_method 'cluster'  --prune_thr 0.15> ./logs/ada_Flickr_iso_gcn2gcn.log 2>&1 &

'''Formal'''
'''Flickr'''
# Basic
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/basic_Flickr_atk_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/basic_Flickr_pru_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 1 > ./logs/basic_Flickr_iso_gcn2gcn.log 2>&1 &
# Basic (prune 0.3)
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.3 --dis_weight 1 --device_id 0 > ./logs/basic_Flickr_pru_gcn2gcn_pr03.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.3 --dis_weight 1 --device_id 1 > ./logs/basic_Flickr_iso_gcn2gcn_pr03.log 2>&1 &
# Unno (prune 0.2)
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/unno_Flickr_atk_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/unno_Flickr_pru_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 1 > ./logs/unno_Flickr_iso_gcn2gcn.log 2>&1 &
# Unno (prune 0.3)
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.3 --dis_weight 1 --device_id 0 > ./logs/unno_Flickr_atk_gcn2gcn_pr03.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.3 --dis_weight 1 --device_id 0 > ./logs/unno_Flickr_pru_gcn2gcn_pr03.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.3 --dis_weight 1 --device_id 1 > ./logs/unno_Flickr_iso_gcn2gcn_pr03.log 2>&1 &
# Adap (prune 0.2)
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/ada_Flickr_atk_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/ada_Flickr_pru_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 1 --device_id 1 > ./logs/ada_Flickr_iso_gcn2gcn.log 2>&1 &
# Adap (tar dis 0)
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 0 --device_id 0 > ./logs/ada_Flickr_atk_gcn2gcn_dis0.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 0 --device_id 0 > ./logs/ada_Flickr_pru_gcn2gcn_dis0.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 0 --device_id 1 > ./logs/ada_Flickr_iso_gcn2gcn_dis0.log 2>&1 &
# Adap (prune 0.3)
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.3 --dis_weight 1 --device_id 0 > ./logs/ada_Flickr_atk_gcn2gcn_pr03.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.3 --dis_weight 1 --device_id 0 > ./logs/ada_Flickr_pru_gcn2gcn_pr03.log 2>&1 &
nohup python run.py --dataset 'Flickr' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.3 --dis_weight 1 --device_id 1 > ./logs/ada_Flickr_iso_gcn2gcn_pr03.log 2>&1 &
'''Reddit2'''
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/basic_Reddit2_atk_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 1 > ./logs/basic_Reddit2_pru_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 0 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 2 > ./logs/basic_Reddit2_iso_gcn2gcn.log 2>&1 &

# Unno (prune 0.3)
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/unno_Reddit2_atk_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 1 > ./logs/unno_Reddit2_pru_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'none' --prune_thr 0.2 --dis_weight 1 --device_id 2 > ./logs/unno_Reddit2_iso_gcn2gcn.log 2>&1 &

nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'none' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 1 --device_id 0 > ./logs/ada_Reddit2_atk_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'prune' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 1 --device_id 1 > ./logs/ada_Reddit2_pru_gcn2gcn.log 2>&1 &
nohup python run.py --dataset 'Reddit2' --model 'GCN' --test_model 'GCN' --attack_method 'Basic' --defense_mode 'isolate' --homo_loss_weight 1 --selection_method 'cluster' --prune_thr 0.2 --dis_weight 1 --device_id 2 > ./logs/ada_Reddit2_iso_gcn2gcn.log 2>&1 &