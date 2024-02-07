#!/bin/bash
DEBUG='' # ['-d','']
N_FOLD=5
ACT_SIZE=3
N_SAMPLES=10000

echo "settings: ${DEBUG} N_FOLD=${N_FOLD} ACT_SIZE=${ACT_SIZE} N_SAMPLES=${N_SAMPLES}"
# repeat for all 5 iterations of the dataset
for yX in {2..2}
    do
    for cX in {2..2}
        do
        for cy in {0..4}
        do
            gpu=$(( (cy+4) % 8 ))
            echo "Dataset with     cy=${cy}    cX=${cX}     yX=${yX}"    
            # run the model
            nohup python3 ../fit_DL_model.py ${DEBUG} --data_dir dataset/toybrains_n${N_SAMPLES}_lblmidr-consite_cy${cy}-cX${cX}-yX${yX} -b 128 -k ${N_FOLD} --gpus ${gpu} --final_act_size $ACT_SIZE -n c1-f${ACT_SIZE}  &> _nohup/nohup_c1-f${ACT_SIZE}_cy${cy}-cX${cX}-yX${yX}.out &
            # wait for a second to start the next iteration
            sleep 1
        done
    done
done