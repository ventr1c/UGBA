models=(GCN GraphSage)
sizes=(80 160 240 320 400 480)
defense_modes=(none prune isolate)
for defense_mode in ${defense_modes[@]};
do 
    for vs_size in ${sizes[@]};
    do
        for model in ${models[@]};
        do
            python -u run_bkd_baseline.py \
                --prune_thr=0.8\
                --attack_method=Rand_Gene\
                --vs_size=${vs_size}\
                --test_model=${model}\
                --defense_mode=${defense_mode}\
                --epochs=500
        done    
    done
done

for defense_mode in ${defense_modes[@]};
do 
    for vs_size in ${sizes[@]};
    do
        for model in ${models[@]};
        do
            python -u run_bkd_baseline.py \
                --prune_thr=0.8\
                --attack_method=Rand_Samp\
                --vs_size=${vs_size}\
                --test_model=${model}\
                --defense_mode=${defense_mode}\
                --epochs=500
        done    
    done
done