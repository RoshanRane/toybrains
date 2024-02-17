# standard python packages
import os, sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm.auto import tqdm
import joblib 
import re
from datetime import datetime

# if parent 'Toybrains' directory is not in path, add it
if os.path.abspath('../../') not in sys.path:
    sys.path.append(os.path.abspath('../../'))

from create_toybrains import ToyBrainsData
from utils.vizutils import *
from utils.configutils import *


# for each dataset find the holdout data that are (a) an equivalent dataset with no conf signal and (b) an equivalent dataset with no true signal 
def generate_baseline_results(dataset, holdout_data=None,
                              input_feature_sets=["attr_all", "attr_subsets", "cov_all", "attr_supset"],
                              output_labels=["lbls"], 
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
            model_name=model_name, model_params=model_params,
            holdout_data=holdout_data,
            compute_shap=compute_shap,
            outer_CV=n_trials, n_jobs=n_jobs,
            metrics=metrics,
            debug=False,
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

    # get debug mode as an command line arg
    DEBUG = False
    if len(sys.argv)>1:
        DEBUG = True if sys.argv[1]=='-d' else False

    time = datetime.now()

    ### Config
    MODELS = [
        ('LR',{}),
        # ('LR',{'penalty':'l1', 'C':1.0, 'solver':'saga'}),
        # ('SVM',{}),
        # ('SVM',{'kernel':'linear'}),
        # ('RF',{}),
        # ('MLP',{}),
        ]
    MODEL_PARAMS = {} 
    COMPUTE_SHAP = True

    INPUT_FEATURE_SETS = [
        "images", 
        # for estimating configured associations (CA)
        "attr_all", "attr_subsets", 
        #  "cov_all", 
    ]
    # ### Load the baseline results
    basefilename = 'lblmidr-consite'
    N_SAMPLES  = 5000
    VERBOSE = 0
    N_TRIALS = 5
    N_JOBS = 5

    if DEBUG:
        N_SAMPLES = 100
        VERBOSE = 2
        N_TRIALS = 2 if N_TRIALS>2 else N_TRIALS
        N_JOBS = 1

    
    datasets = glob(f"dataset/*n{N_SAMPLES}_*{basefilename}*/")
    # create tuples of each dataset and its unbiased test data (Aconf=0)
    dataset_tuples = []
    for dataset in datasets:
        dataset_noconf = re.sub('cX...', 'cX000', re.sub('cy...','cy000', dataset))
        assert os.path.exists(dataset_noconf), f"Could not find noconf dataset {dataset_noconf} for {dataset}"
        # force the Xy relation to be the max for the extreme test dataset
        dataset_noconf_ext = re.sub('yX...', 'yX100', dataset_noconf)
        assert os.path.exists(dataset_noconf_ext), f"Could not find noconf dataset {dataset_noconf_ext} for {dataset}"

        dataset_nosignal = re.sub('yX...','yX000', dataset)
        assert os.path.exists(dataset_nosignal), f"Could not find nosignal dataset {dataset_nosignal} for {dataset}"
        # force the X<-c->y relation to be the max for the extreme test dataset
        dataset_nosignal_ext = re.sub('cX...', 'cX100', re.sub('cy...','cy100', dataset_nosignal))
        assert os.path.exists(dataset_nosignal_ext), f"Could not find nosignal dataset {dataset_nosignal_ext} for {dataset}"
        
        dataset_tuples.append((dataset, 
                                {'no-conf': dataset_noconf,   'no-conf-ext': dataset_noconf_ext,
                                    'no-true': dataset_nosignal, 'no-true-ext': dataset_nosignal_ext}))
            
    dataset_model_tuples = [((m, m_params),(d, hold_d)) for m, m_params in MODELS for d, hold_d in dataset_tuples]
    
    print(f"On {len(datasets)} datasets, running {len(MODELS)} models x {N_TRIALS} trials  = {len(dataset_model_tuples)*N_TRIALS} total runs")

    # convert to a generator for tqdm
    def get_dataset_model_tuples(tups):
        i = 0     
        for i in range(len(tups)):         
            yield tups[i]

    #  generate the baseline results on generated datasets using parallel processes
    bl_results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(
            generate_baseline_results)(
                dataset, holdout_data=holdout_data,
                input_feature_sets=INPUT_FEATURE_SETS, 
                model_name=model_name, model_params=model_params,
                output_labels=["lbls"],
                compute_shap=COMPUTE_SHAP,
                metrics=['balanced_accuracy', 'r2', 'roc_auc'], 
                n_trials=N_TRIALS, n_jobs=N_JOBS,
                n_samples=N_SAMPLES, verbose=VERBOSE) \
 for (model_name, model_params), (dataset, holdout_data) in tqdm(get_dataset_model_tuples(dataset_model_tuples)))

    # print the total runtime in 
    print("Total runtime:", strfdelta(datetime.now()-time, "{days} days, {hours} Hrs: {minutes} min: {seconds} secs"))
# %%
