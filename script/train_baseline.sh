models=(GCN GraphSage GAT)
defense_modes=(none prune isolate)
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_bkd_baseline.py \
#             --prune_thrd=0.2\
#             --attack_method=Rand_Gene\
#             --vs_size=40\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --epochs=200\
#             --dataset=Pubmed
#     done    
# done

for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        python -u run_GTA.py \
            --prune_thr=0.2\
            --vs_size=40\
            --test_model=${model}\
            --defense_mode=${defense_mode}\
            --epochs=200\
            --dataset=Pubmed
    done    
done