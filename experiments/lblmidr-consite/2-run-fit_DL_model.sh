#!/bin/bash
DEBUG='' # ['-d','']
N_FOLD=3
N_SAMPLES=5000
N_GPUS=8
echo "global settings: ${DEBUG} N_FOLD=${N_FOLD} N_SAMPLES=${N_SAMPLES}"
# repeat for all 5 iterations of the dataset

gpu=0
for yX in '025' '050'; 
    do
    for cX in '100'; 
        do
        cy=$cX
        for ACT_SIZE in '003' '064' '256';
            do
            echo "Dataset with     cy = ${cy}    cX = ${cX}     yX = ${yX}      &  ACT_SIZE = ${ACT_SIZE}        on gpu ${gpu}"    
            # run the model
            nohup python3 ../fit_DL_model.py ${DEBUG} --data_dir dataset/toybrains_n${N_SAMPLES}_lblmidr-consite_cy${cy}-cX${cX}-yX${yX} -b 128 -k ${N_FOLD} --gpus ${gpu} --final_act_size $ACT_SIZE -n c1-f${ACT_SIZE}  &> _nohup/nohup_c1-f${ACT_SIZE}_cy${cy}-cX${cX}-yX${yX}.out &
            # wait for a second to start the next iteration
            sleep 1
            done
        # increment GPU
        gpu=$(( (gpu+1) % N_GPUS ))
        sleep 5
        done
    sleep 15 # wait for 15 seconds to start the next yX
    done
    
