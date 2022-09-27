models=(GCN GraphSage)
sizes=(80 160 240 320 400 480)
defense_modes=(none prune isolate)
for defense_mode in ${defense_modes[@]};
do 
    for vs_size in ${sizes[@]};
    do
        for model in ${models[@]};
        do
            for seed in {15..18}
            do
                python -u run_bkd_baseline.py \
                    --prune_thrd=0.3\
                    --vs_size=${vs_size}\
                    --test_model=${model}\
                    --defense_mode=${defense_mode}\
                    --seed=${seed} \
                    --epochs=1000
            done
        done    
    done
done