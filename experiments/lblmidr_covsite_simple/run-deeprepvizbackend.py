# standard python packages
import os, sys, shutil
from glob import glob
import numpy as np
import pandas as pd

PARENT_REPO = os.path.abspath("../")
if PARENT_REPO not in sys.path: sys.path.append(PARENT_REPO)
from utils.DLutils import *
from utils.vizutils import *

DEEPREPVIZ_REPO = os.path.abspath("../../Deep-confound-control-v2/")
if DEEPREPVIZ_REPO not in sys.path: sys.path.append(DEEPREPVIZ_REPO)
from DeepRepVizBackend import DeepRepVizBackend




if __name__ == "__main__":
    

    for DATASET_UNIQUE_ID in ['t0', 't1', 't2', 't3', 't4']:
        attributes_table = f"../dataset/toybrains_n10000_lblmidr_covsite_{DATASET_UNIQUE_ID}/toybrains_n10000_lblmidr_covsite_{DATASET_UNIQUE_ID}.csv"
        df_attrs = pd.read_csv(attributes_table)

        ID_col = 'subjectID'
        label = 'lbl_lesion'

        drv_backend = DeepRepVizBackend(        
                        conf_table=df_attrs,
                        ID_col=ID_col, label_col=label,
                        best_ckpt_by='loss', best_ckpt_metric_should_be='min',
                        debug=False)

        # Make a list of all models (different model architecture or different runs/versions of the same model architecture)
        # trained on the current dataset
        logdirs = sorted([logdir for logdir in glob(f"log/toybrains-{DATASET_UNIQUE_ID}*/trial*/deeprepvizlog/") if 'debug' not in logdir])
        logdirs

        # ### Load the trained model log dirs
        for logdir in logdirs:
            drv_backend.load_log(logdir)
        # the logs are loaded in drv_backend.deeprepvizlogs and can be printed
        drv_backend._pprint_deeprepvizlogs()

        # ### downsample the activations to 3D
        drv_backend.downsample_activations(overwrite=False)

        # you can also ask the backend to create a table compatible with the DeepRepViz v1 frontend for each log individually
        for logdir in logdirs:
            df = drv_backend.convert_log_to_v1_table(log_key=logdir, overwrite=False, 
                                                    unique_name=logdir.split('/')[-4])

        # ### Compute metrics
        # The backend can be asked to compute metrics for specific logdirs in the deeprepvizlogs
        results = []
        drv_backend.debug = False
        for i, logdir in enumerate(logdirs):
            # you can first verify check if the metrics have been computed and stored already 
            metrics = ['dcor', 'mi', 'con', 'costeta', 'r2']
            existing_metrics = drv_backend.get_metrics(logdir, ckpt_idx='best')
            if existing_metrics is not None:
                metrics = [m for m in metrics if m not in existing_metrics]
                print(f"Skipping {list(existing_metrics.keys())} for {logdir}. As they have already been computed.")
            # it will compute it and store it in the metametadata.json file of the logdirs
            result = drv_backend.compute_metrics(log_key=logdir,
                                                metrics=metrics,
                                                #   covariates=['lbl_lesion','cov_site', 'brain-int_fill','shape-midr_curv', 'shape-midr_vol-rad'], 
                                                ckpt_idx='best')
            # the computed metrics is also additionally returned as a dictionary
            # display(result)
            # results.append(result)