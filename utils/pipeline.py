# standard python packages
import os
import random
from glob import glob
from datetime import datetime
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np
from pathlib import Path

import monai
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger

# add custom imports
from utils.dataset import generate_dataset
from utils.DLutils import (
    get_dataset_loaders,
    ToyBrainsDataset, LightningModel,
    PyTorchMLP, LogisticRegression, viz_batch
)
from utils.baseline import get_table_loader, run_lreg
from utils.sklearn import get_reduc_loader, run_logistic_regression

# functions

# run toybrains dataset baseline
def run_baseline_pipeline(file_path, setting=None, debug=False):
    ''' run baseline pipeline 
    
    there are three main:
    1. baseline on attributes -> labels, cov
    2. supervised learning on images -> labels
    3. unsupervised learning on images -> labels
    
    PARAMETERS
    ----------
    file_path : string
        toybrains dataset file path
        it supports absoulte or relative path
        
    setting : dictionary
        setting with the style of data_dict
    
    debug : boolean
        debug mode
        
    NOTE
    ----
    all the output saved in results directory with the name of toybrains dataset directory name
    '''
    
    start_time = datetime.now()
    file_split = file_path.split('/')
    DATA_DIR, DATA_FILE = file_split[-2], file_split[-1]
    DATA_N = DATA_FILE[11:][:-4]
    print(f'{"="*70}\ntime: {start_time}\nRunning Baseline on a file: {DATA_DIR}/{DATA_FILE}\n{"="*70}\n')
    
    # create the folder in which to save the results
    if debug:
        # (TODO) There is no difference on debug mode
        os.system("rm -rf results/debug_run 2> /dev/null")
        print(f'DATA DIR : {DATA_DIR}\nDATA N   : {DATA_N}\n')
        OUT_DIR = Path.cwd() / 'results' / 'debug_run' / start_time.strftime("%Y%m%d-%H%M")
        print(f'Save in {OUT_DIR}')
    else:
        OUT_DIR = Path.cwd() / 'results' / DATA_DIR / start_time.strftime("%Y%m%d-%H%M")
    OUT_DIR.mkdir(parents = True, exist_ok = True)
    
    # DEFAULT SETTING
    # (TODO) could be merge with setting as dict with below
    options = 'bs' # default, usb
    labels = None
    seed = 42
    debug = False
    num = 0
    
    # ONLY FOR SUPERVISED
    gpu = [1]
    seed = 42
    models = None
    batch_size = 16
    learning_rate = 0.05
    transform = []
    max_epochs = 10
    accelerator = 'gpu'
    device = [2]

    # ONLY FOR UNSUPERVISED
    methods = None
    n_components = None
    
    # (TODO) merge with above? it depends on how to pass a file_path
    # ? Option to change the setting if needed
    # ? Which can be an setting options?
    # data_dict can contains other setting
    # OR other options, pass it with zip() in upper loop on csv_path_list
    # if DATA_DIR in setting.keys():
    #     SETTING = setting[DATA_DIR]
    #     # print(SETTING)
    #     if 'options' in SETTING.keys():
    #         options = SETTING['options']
    
    # (TODO) ?Applying Parrallel
    
    # run baseline
    if 'b' in options:
        
        run_baseline(
            raw_csv_path = file_path,
            DATA_DIR = DATA_DIR,
            DATA_N = DATA_N,
            OUT_DIR = OUT_DIR,
            seed = seed,
            debug = debug
        )
        num += 1 # fixed number

    # run supervised learning
    if 's' in options:
        
        run_supervised(
            raw_csv_path = file_path,
            DATA_DIR = DATA_DIR,
            DATA_N = DATA_N,
            OUT_DIR = OUT_DIR,
            labels = labels,
            models = models,
            batch_size = batch_size,
            learning_rate = learning_rate,
            transform = transform,
            max_epochs = max_epochs,
            accelerator = accelerator,
            device = device,
            seed = seed,
            debug = debug,
        )
        num += 1 # counting needed
        
    # run unsupervised learning
    # (TODO) solve the error
    if 'u' in options:
        
        run_unsupervised(
            raw_csv_path = file_path,
            DATA_DIR = DATA_DIR,
            DATA_N = DATA_N,
            OUT_DIR = OUT_DIR,
            labels = labels,
            methods = methods,
            n_components = n_components,
            seed = seed,
            debug = debug,
        )
        num += 1 # counting needed
    
    print(f'running a total of {num} different settings of models')
    
    df = pd.concat([pd.read_csv(csv) for csv in glob(f"{str(OUT_DIR)}/run_*.csv")], ignore_index=True)
    df = df.sort_values(["dataset", "inp", "out", "type", "model"]) # sort
    # (TODO) Reorder columns for readability
    df.to_csv(f"{str(OUT_DIR)}/run.csv", index=False)
    
    # delete the temp csv files
    os.system(f"rm {str(OUT_DIR)}/run_*.csv")
    
    runtime = str(datetime.now()-start_time).split(".")[0]
    print(f'TOTAL RUNTIME: {runtime}')
    

# run supervised learning on images
def run_supervised(
    raw_csv_path,
    DATA_DIR,
    DATA_N,
    OUT_DIR,
    labels = None,
    models = None,
    batch_size = 16,
    learning_rate = 0.05,
    transform = [],
    max_epochs = 10,
    accelerator = "gpu",
    device = [1],
    seed = 42,
    debug = False
):
    ''' run supervised learning 
    
    It currently support only  binary classification label:
        'lblbin_shp', 'lblbin_shp-vol', 'lblbin_shp-vent', 'Sex'
    
    PARAMETERS
    ----------
    raw_csv_path : string
        toybrains_n*.csv path
    DATA_DIR : string
        raw_csv_path input directory
    DATA_N : string
        n samples
    OUT_DIR : pathlib.PosixPath
        run.csv output directory
    labels : (None | list), default : None
        list of label
        None use default embedded labels
    models : (None | dictionary), default : None
        list of model
        None use default embedded models
    batch_size : int, default : 16
        batch size
    learning_rate : float, default : 0.05
        learning rate
    transform : [], default : []
        transform if needed
    max_epochs : int, default : 16
        max epochs
    accelerator : string, default : gpu
        accelerator for lightning trainer
    device : (list | int), default : [1]
        device for lightning trainer
    seed : int, default : 42
        random state for reproducibility
    debug : boolean, default : False
        debug mode
    '''
    print('run supervised learning')
    
    # set random seed
    torch.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # set the seed for Lightning
    L.seed_everything(seed)
    
    # (TODO) sanity check on raw_csv_path, DATA_DIR, and OUT_DIR
    if debug:
        DATA_DF = pd.read_csv(raw_csv_path)
        print(DATA_DF.info())
    
    label_list = labels if labels else ['lblbin_shp', 'lblbin_shp-vol', 'lblbin_shp-vent', 'cov_sex']
    models = models if models else dict(
        MLP = PyTorchMLP(num_features=12288, num_classes=2),
        # below took so long
        # DenseNet = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=3, out_channels=2)
    )
    
    # (TODO) Parallel needed?
    # choose a targert label among the available columns in the table
    for label in label_list:
        # split the dataset for training, validation, and test from raw dataset
        df_train, df_val, df_test = generate_dataset(raw_csv_path, label, seed)
        print(f" Training data split = {len(df_train)} \n Validation data split = {len(df_val)} \n Test data split = {len(df_test)}")
        
        # prepare the dataloader
        train_loader, val_loader, test_loader = get_dataset_loaders(
            data_split_dfs=[df_train, df_val, df_test],
            data_dir=DATA_DIR,
            batch_size=batch_size, shuffle=True,
            num_workers=0, transform=transform)
        
        if debug:
            viz_batch(train_loader, title="Training images", debug=True)
            viz_batch(val_loader, title="Validation images", debug=True)
            viz_batch(test_loader, title="Test images", debug=True)

        common = {
            "dataset" : DATA_DIR,
            "type" : "supervised",
            "n_samples" : DATA_N,
            "inp" : "images",
            "out" : label,
        }
            
        # (TODO) add the accuracy when predicting the most frequent class label
        # bottle neck
#         train_counter = Counter()
#         for _, labels in train_loader: train_counter.update(labels.tolist())
#         train_majority_class = train_counter.most_common(1)[0]
#         train_baseline_acc = train_majority_class[1] / sum(train_counter.values())
        
#         val_counter = Counter()
#         for _, labels in val_loader: val_counter.update(labels.tolist())
#         val_majority_class = val_counter.most_common(1)[0]
#         val_baseline_acc = val_majority_class[1] / sum(val_counter.values())
        
#         test_counter = Counter()
#         for _, labels in test_loader: test_counter.update(labels.tolist())
#         test_majority_class = test_counter.most_common(1)[0]
#         test_baseline_acc = test_majority_class[1] / sum(test_counter.values())
        
#         baseline = {
#             "train_majority_class" : train_majority_class,
#             "train_baseline_acc" : train_baseline_acc,
#             "val_majority_class" :val_majority_class,
#             "val_baseline_acc" :val_baseline_acc,
#             "test_majority_class" : test_majority_class,
#             "test_baseline_acc" : test_baseline_acc,
#         }
        
#         common.update(baseline)
        
        # (TODO) Parallel?
        for model_name, pytorch_model in models.items():
            # set lightning model
            lightning_model = LightningModel(model=pytorch_model, learning_rate=learning_rate)
            trainer = L.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=device,
                logger=CSVLogger(save_dir=f"{str(OUT_DIR)}/supervised_logs/{label}", name=model_name),
                deterministic=True,
                enable_progress_bar=False
            )
            
            trainer.fit(
                model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
            
            train_acc = trainer.test(dataloaders=train_loader)[0]["accuracy"]
            val_acc = trainer.test(dataloaders=val_loader)[0]["accuracy"]
            test_acc = trainer.test(dataloaders=test_loader)[0]["accuracy"]
            
            if debug:
                print(f"Train Acc {train_acc*100:.2f}%"
                      f" | Val Acc {val_acc*100:.2f}%"
                      f" | Test Acc {test_acc*100:.2f}%")
            
            # (TDOO) add loss and other metrics
            result = {
                "model" : model_name,
                "model_config" : pytorch_model,
                # "epoch" : max_epochs,
                # "batch_size" : batch_size,
                "train_metric" : train_acc,
                "val_metric" : val_acc,
                "test_metric" : test_acc,
            }
            result.update(common)
            
            df = pd.DataFrame([result])
            df.to_csv(f"{str(OUT_DIR)}/run_sup_{label}_{model_name}.csv", index=False)

# run unsupervised learning on images
def run_unsupervised(
    raw_csv_path,
    DATA_DIR,
    DATA_N,
    OUT_DIR,
    labels = None,
    methods = None,
    n_components = None,
    seed = 42,
    debug = False,
):
    ''' run unsupervised learning
    
    PARAMETERS
    ----------
    raw_csv_path : string
        toybrains_n*.csv path
    DATA_DIR : string
        raw_csv_path input directory
    DATA_N : string
        n samples
    OUT_DIR : pathlib.PosixPath
        run.csv output directory
    labels : (None | list), default : None
        list of label
        None use default embedded labels
    methods : (None | list), default : None
        list of dimensionality reduction method
    n_components : (None | list), default : None
        list of integer
    '''
    print('run unsupervised learning')
    
    common = {
        "dataset" : DATA_DIR,
        "type" : "unsupervised",
        "n_samples" : DATA_N,
    }
    
    # set target label
    label_list = labels if labels else ['lblbin_shp', 'lblbin_shp-vol', 'lblbin_shp-vent', 'cov_sex']
    # set dimensionality reduction methods
    methods = methods if methods else ['PCA'] # TODO 'ICA', 'MDS', ...
    # n components
    n_components = n_components if n_components else [100]
    # parallel using setting dictionary
    for label in label_list:
        print(f'{label} generate dataset')
        dataset = generate_dataset(raw_csv_path, label, seed)
        for method in methods:
            print(f'{method} using')
            
            for n in n_components:
                print(f'load the loader in {n} with {DATA_DIR}')
                data = get_reduc_loader(dataset=dataset, data_dir=DATA_DIR, method=method, n_components=n, seed=seed)
                # run logistic regression
                if debug:
                    print(f"Label : {label} use Dimensionality reduction method ({method}) images into {n} point")
                print('run')
                metric, pipe = run_logistic_regression(data)
                train_metric, val_metric, test_metric = metric
                model_name = f'{methods}-logistic_regression'
                setting = {
                    "inp" : f'images_to_{n}_components',
                    "out" : label,
                    "model" : model_name,
                    "model_config" : pipe,
                    "train_metric" : train_metric,
                    "val_metric" : val_metric,
                    "test_metric" : test_metric
                }
                result.update(common)
                
                df = pd.DataFrame([result])
                df.to_csv(f"{str(OUT_DIR)}/run_usp_{label}_{n}_{model_name}.csv", index=False)