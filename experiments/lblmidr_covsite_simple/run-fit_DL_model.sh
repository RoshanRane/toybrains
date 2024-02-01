nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t0 --gpus 0 --final_act_size 3 -n "_cls1-fea3" &> nohup_0-3.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t1 --gpus 1 --final_act_size 3 -n "_cls1-fea3" &> nohup_1-3.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t2 --gpus 2 --final_act_size 3 -n "_cls1-fea3" &> nohup_2-3.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t3 --gpus 3 --final_act_size 3 -n "_cls1-fea3" &> nohup_3-3.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t4 --gpus 4 --final_act_size 3 -n "_cls1-fea3" &> nohup_4-3.out &


nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t0 --gpus 5 --final_act_size 64 -n "_cls1-fea64" &> nohup_0-64.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t1 --gpus 6 --final_act_size 64 -n "_cls1-fea64" &> nohup_1-64.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t2 --gpus 7 --final_act_size 64 -n "_cls1-fea64" &> nohup_2-64.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t3 --gpus 0 --final_act_size 64 -n "_cls1-fea64" &> nohup_3-64.out &
nohup python3 fit_DL_model.py --data_dir dataset/toybrains_n10000_lblmidr_covsite_t4 --gpus 1 --final_act_size 64 -n "_cls1-fea64" &> nohup_4-64.out &