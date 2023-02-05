models=(GCN GraphSage GAT)
# isolate means the Prune+LD defense method
defense_modes=(none prune isolate)

# Cora
for defense_mode in ${defense_modes[@]};
do 
    for model in ${models[@]};
    do
        python -u run_adaptive.py \
            --prune_thr=0.1\
            --dataset=Cora\
            --homo_loss_weight=50\
            --vs_number=10\
            --test_model=${model}\
            --defense_mode=${defense_mode}\
            --selection_method=cluster_degree\
            --homo_boost_thrd=0.5\
            --epochs=200\
            --trojan_epochs=400
    done    
done

# # Pubmed
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --prune_thr=0.2\
#             --dataset=Pubmed\
#             --homo_loss_weight=100\
#             --vs_number=40\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=cluster_degree\
#             --homo_boost_thrd=0.5\
#             --epochs=200\
#             --trojan_epochs=2000
#     done    
# done

# # Flickr
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --prune_thr=0.4\
#             --dataset=Flickr\
#             --hidden 64 \
#             --homo_loss_weight=100\
#             --vs_number=80\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=cluster_degree\
#             --homo_boost_thrd=0.8\
#             --epochs=200\
#             --trojan_epochs=400
#     done    
# done

# # OGBN-Arixv
# for defense_mode in ${defense_modes[@]};
# do 
#     for model in ${models[@]};
#     do
#         python -u run_adaptive.py \
#             --prune_thr=0.8\
#             --dataset=ogbn-arxiv\
#             --homo_loss_weight=200\
#             --vs_number=160\
#             --test_model=${model}\
#             --defense_mode=${defense_mode}\
#             --selection_method=cluster_degree\
#             --homo_boost_thrd=0.8\
#             --epochs=800\
#             --trojan_epochs=800
#     done    
# done
