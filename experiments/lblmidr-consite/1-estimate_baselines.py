# standard python packages
import os, sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
import joblib 
import re
from datetime import datetime

# if parent 'Toybrains' directory is not in path, add it
if os.path.abspath('../../') not in sys.path:
    sys.path.append(os.path.abspath('../../'))

from create_toybrains import ToyBrainsData
from utils.vizutils import *
from utils.configutils import *
import argparse


# for each dataset find the holdout data that are (a) an equivalent dataset with no conf signal and (b) an equivalent dataset with no true signal 
def generate_baseline_results(dataset, holdout_data=None,
                              input_feature_sets=["attr_all", "attr_subsets", "cov_all", "attr_supset"],
                              output_labels=["lbls"], 
                              conf_ctrl={},
                              model_name='LR', model_params={},
                              metrics=['r2'],
                              compute_shap=True, n_jobs=-1,
                              n_trials=5, n_samples=1000, verbose=0):    
    # get the related config file
    dataset_unique_id = dataset.rstrip('/').split('/')[-1].split(f'n{n_samples}_')[-1]
    config_file = glob(f"configs/*{dataset_unique_id}*.py")
    assert len(config_file)==1, f"couldn't find the config file used to generate the dataset with unique ID '{dataset_unique_id}'. \nFound {config_file}"
    config_file = config_file[0]

    # init the ToyBrainsData instance
    toy = ToyBrainsData(config=config_file)
    toy.load_generated_dataset(dataset)
    bl_result = toy.fit_contrib_estimators(
                            input_feature_sets=input_feature_sets,
                            output_labels=output_labels, 
                            conf_ctrl=conf_ctrl,
                            model_name=model_name, model_params=model_params,
                            holdout_data=holdout_data,
                            compute_shap=compute_shap,
                            outer_CV=n_trials, n_jobs=n_jobs,
                            metrics=metrics,
                            verbose=verbose)
    
    return bl_result

# test case
# dfi = generate_baseline_results(dataset_tuples[0][0], holdout_data=dataset_tuples[0][1], 
#                                 input_feature_sets=["attr_all"], 
#                                 output_labels=["lbls"],
#                                 metrics=['balanced_accuracy', 'r2', 'roc_auc'], n_trials=2, 
#                                 model_name=MODEL_NAME, model_params=MODEL_PARAMS,
#                                 compute_shap=COMPUTE_SHAP,
#                                 n_samples=N_SAMPLES, verbose=2)
# dfi

# %%
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

# %%

# run the full script
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    DEBUG = False
    parser.add_argument('-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--njobs', type=int, default=5, help='Number of jobs to run in parallel')

    args = parser.parse_args()
    DEBUG = args.d
    N_JOBS = args.njobs

    time = datetime.now()

    ### Config
    MODELS = [
        # ('LR',{}),
        # ('LR',{'penalty':'l1', 'C':1.0, 'solver':'liblinear'}),
        # ('SVM',{'kernel':'linear'}),
        # ('SVM',{}),
        ('RF',{}),
        # ('MLP',{}),
        ]
    COMPUTE_SHAP = True

    INPUT_FEATURE_SETS = [
        "images", 
        # for estimating configured associations (CA)
        "attr_all", "attr_subsets", 
        #  "cov_all", 
    ]

    CONF_CTRL = {
        "upsample"  : ["cov_site"],
        "resample"  : ["cov_site"],
        "downsample": ["cov_site"],
    }
    INCLUDE_DATASET_EXTREMES = True

    # ### Load the baseline results
    basefilename = 'lblmidr-consite'
    N_SAMPLES  = 5000
    N_TRIALS = 5
    VERBOSE = 0

    if DEBUG:
        N_SAMPLES = 100
        VERBOSE = 2
        N_TRIALS = 2 if N_TRIALS>2 else N_TRIALS
        INCLUDE_DATASET_EXTREMES = False

    
    datasets = glob(f"dataset/*n{N_SAMPLES}_*{basefilename}*/")

    ### for each dataset find the holdout data that are (a) an equivalent dataset with no conf signal and (b) an equivalent dataset with no true signal 
    dataset_tuples = []
    test_suffix = '_test'
    test_nsamples = 1000

    for dataset in datasets:
        dataset = dataset.rstrip('/')
        dataset_test = re.sub(f'_n{N_SAMPLES}_', f'_n{test_nsamples}_', dataset) + test_suffix
        dataset_noconf = re.sub('cX...', 'cX000', re.sub('cy...','cy000', dataset_test))
        assert os.path.exists(dataset_noconf), f"Could not find noconf dataset {dataset_noconf} for {dataset}"

        dataset_nosignal = re.sub('yX...','yX000', dataset_test) 
        assert os.path.exists(dataset_nosignal), f"Could not find nosignal dataset {dataset_nosignal} for {dataset}"
        
        ood_test_datasets = {'no-conf': dataset_noconf, 'no-true': dataset_nosignal}

        if INCLUDE_DATASET_EXTREMES:
            # also force the Xy relation to be the max
            dataset_noconf_ext = re.sub('yX...', 'yX100', dataset_noconf)
            assert os.path.exists(dataset_noconf_ext), f"Could not find noconf dataset {dataset_noconf_ext} for {dataset}"
            # also force the X<-c->y relation to be the max
            dataset_nosignal_ext = re.sub('cX...', 'cX100', re.sub('cy...','cy100', dataset_nosignal))
            assert os.path.exists(dataset_nosignal_ext), f"Could not find nosignal dataset {dataset_nosignal_ext} for {dataset}"

            ood_test_datasets.update({'no-conf_ext': dataset_noconf_ext, 'no-true_ext': dataset_nosignal_ext})

        dataset_tuples.append((dataset, ood_test_datasets))
            
    dataset_model_tuples = [((m, m_params),(d, hold_d)) for m, m_params in MODELS for d, hold_d in dataset_tuples]
    ndatasets = len(dataset_model_tuples)

    if DEBUG:
        if ndatasets > 10:
            dataset_model_tuples = [dataset_model_tuples[0], dataset_model_tuples[ndatasets//2], dataset_model_tuples[-1]]
        # delete previous baseline results
        print(f"Datasets:")
        for (_,_),(dataset,_) in dataset_model_tuples:
            os.system(f"rm -rf dataset/{dataset}/baseline_results/*")
            print(dataset)

    print(f"[parallel jobs] On {len(dataset_model_tuples)} datasets, running {len(MODELS)} models ({[f'{m}({p})' for m,p in MODELS]}) x {N_TRIALS} trials & {len(CONF_CTRL)} conf. ctrl methods \
  = {len(dataset_model_tuples)*N_TRIALS*(len(CONF_CTRL)+1)} total runs")            
    
    #  generate the baseline results on generated datasets using parallel processes
    bl_results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(
            generate_baseline_results)(
                dataset, holdout_data=holdout_data,
                input_feature_sets=INPUT_FEATURE_SETS, 
                conf_ctrl=CONF_CTRL,
                model_name=model_name, model_params=model_params,
                output_labels=["lbls"],
                compute_shap=COMPUTE_SHAP,
                metrics=['balanced_accuracy', 'r2', 'roc_auc', 'adjusted_mutual_info_score'], 
                n_trials=N_TRIALS, n_jobs=N_JOBS,
                n_samples=N_SAMPLES, verbose=VERBOSE) \
 for (model_name, model_params), (dataset, holdout_data) in tqdm(dataset_model_tuples, total=ndatasets))

    # print the total runtime in 
    print("Total runtime:", strfdelta(datetime.now()-time, "{days} days, {hours} Hrs: {minutes} min: {seconds} secs"))
# %%
