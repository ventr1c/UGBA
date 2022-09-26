models=(GCN GraphSage GAT)
defense_modes=(none prune isolate)
for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        python -u run_adaptive.py \
            --prune_thrd=0.3\
            --dataset=Pubmed\
            --homo_loss_weight=0\
            --vs_size=40\
            --test_model=${model}\
            --defense_mode=${defense_mode}
    done    
done

for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        python -u run_adaptive.py \
            --prune_thrd=0.3\
            --dataset=Pubmed\
            --homo_loss_weight=100\
            --vs_size=40\
            --test_model=${model}\
            --defense_mode=${defense_mode}
    done    
done

models=(GCN GraphSage)
for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        python -u run_adaptive.py \
            --prune_thrd=0.8\
            --dataset=ogbn-arxiv\
            --homo_loss_weight=0.0\
            --vs_size=160\
            --test_model=${model}\
            --defense_mode=${defense_mode}
    done    
done

for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        python -u run_adaptive.py \
            --prune_thrd=0.8\
            --dataset=ogbn-arxiv\
            --homo_loss_weight=100\
            --vs_size=160\
            --test_model=${model}\
            --defense_mode=${defense_mode}
    done    
done