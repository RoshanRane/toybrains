# standard python packages
import os, sys
from os.path import abspath, join
from glob import glob
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

PARENT_REPO = os.path.abspath("../../")
if PARENT_REPO not in sys.path: sys.path.append(PARENT_REPO)
from utils.DLutils import *
from utils.vizutils import *

DEEPREPVIZ_BACKEND = abspath(join( "../../../Deep-confound-control-v2/", "application/backend/deep_confound_control/core/")) 
assert os.path.isdir(DEEPREPVIZ_BACKEND) and os.path.exists(DEEPREPVIZ_BACKEND+'/DeepRepVizBackend.py'), f"No DeepRepViz repository found in {DEEPREPVIZ_BACKEND}. Add the correct relative path to the backend to the 'DEEPREPVIZ_BACKEND' global variable."
if DEEPREPVIZ_BACKEND not in sys.path:
    sys.path.append(DEEPREPVIZ_BACKEND)
from DeepRepVizBackend import DeepRepVizBackend

DEBUG = False
CREATE_V1_TABLE = False
ID_col = 'subjectID'
label = 'lbl_lesion'
N_SAMPLES = 5000

DATASET_SEARCH_PATH = "dataset/toybrains_n{}_*{}/toybrains_*{}.csv"
LOGDIR_SEARCH_PATH = "log/toybrains_n{}_*{}*/trial*/deeprepvizlog/"

if __name__ == "__main__":
    
    # get the list of datasets present in the current directory's log folder that have DeepRepViz logs
    # also, check if the model training for all log folders have completed (TODO check for checkpoint json file rather than the DeepRepViz-v1-*.csv file
    DATASET_UNIQUE_IDS = sorted([os.path.basename(logdir).split('_')[3] for logdir in glob(LOGDIR_SEARCH_PATH.split('/trial')[0].format(N_SAMPLES, '')) \
    if (len(glob(f"{logdir}/*/deeprepvizlog/DeepRepViz-v1-*.csv"))==len(glob(f"{logdir}/*/deeprepvizlog")))]) 
    
    N_JOBS = len(DATASET_UNIQUE_IDS) if len(DATASET_UNIQUE_IDS)<10 else 10

    if DEBUG: 
        N_JOBS = 1
        DATASET_UNIQUE_IDS = DATASET_UNIQUE_IDS[:2]
        print(f"runnning in DEBUG mode .. \nselecting only {len(DATASET_UNIQUE_IDS)} datasets ..")

    print(f"Found {len(DATASET_UNIQUE_IDS)} datasets with DeepRepViz logs with unique IDs: {DATASET_UNIQUE_IDS}")

    # run each dataset processing in parallel
    def run_backend(dataset_unique_ID, label=label, ID_col=ID_col, 
                    debug=DEBUG, create_v1_table=CREATE_V1_TABLE):

        attributes_table = glob(DATASET_SEARCH_PATH.format(N_SAMPLES, dataset_unique_ID, dataset_unique_ID))
        assert len(attributes_table)==1, f"Found {len(attributes_table)} datasets tables matching the query '{dataset_unique_ID}'"
        df_attrs = pd.read_csv(attributes_table[0])

        drv_backend = DeepRepVizBackend(        
                        conf_table=df_attrs,
                        ID_col=ID_col, label_col=label,
                        best_ckpt_by='loss_test', best_ckpt_metric_should_be='min',
                        debug=debug)

        # Make a list of all models (different model architecture or different runs/versions of the same model architecture)
        # trained on the current dataset
        logdirs = sorted([logdir for logdir in glob(LOGDIR_SEARCH_PATH.format(N_SAMPLES, dataset_unique_ID)) if 'debug' not in logdir])
        if len(logdirs)==0:
            print(f"[WARN] No logdirs found for dataset with query ID '{dataset_unique_ID}'. something fishy???")
            return

        ### Load all the trained model logdirs
        for logdir in logdirs:
            drv_backend.load_log(logdir)

        # the logs are loaded in drv_backend.deeprepvizlogs and can be printed
        if debug:
            print(f"Found {len(logdirs)} logdirs for dataset with query ID '{dataset_unique_ID}'") 
            drv_backend._pprint_deeprepvizlogs()

        # ### downsample the activations to 3D
        drv_backend.downsample_activations(overwrite=False)

        # ### Compute metrics
        # The backend can be asked to compute metrics for specific logdirs in the deeprepvizlogs
        # results = []
        drv_backend.debug = False
        for i, logdir in enumerate(logdirs):
            # you can also ask the backend to create a table compatible with the DeepRepViz v1 frontend for each log individually
            if create_v1_table:
                df = drv_backend.convert_log_to_v1_table(log_key=logdir, overwrite=False, 
                                                        unique_name=logdir.split('/')[-4])
            # you can first verify check if the metrics have been computed and stored already 
            metrics = ['dcor', 'mi', 'con', 'costeta', 'r2']
            existing_metrics = drv_backend.get_metrics(logdir, ckpt_idx='best')
            if existing_metrics is not None:
                metrics = [m for m in metrics if m not in existing_metrics]
                print(f"Skipping {list(existing_metrics.keys())} for {logdir}. As they have already been computed.")

            # it will compute it and store it in the metametadata.json file of the logdirs
            # TODO parallelize this for loop
            result = drv_backend.compute_metrics(log_key=logdir,
                                                metrics=metrics,
                                                #   covariates=['lbl_lesion','cov_site', 
                                                #'brain-int_fill','shape-midr_curv', 'shape-midr_vol-rad'], 
                                                ckpt_idx='best')
            # the computed metrics is also additionally returned as a dictionary
            # display(result)
            # results.append(result)


    # Parallelize the processing of datasets
    joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(
        run_backend)(dataset_unique_ID) for dataset_unique_ID in tqdm(DATASET_UNIQUE_IDS))
        