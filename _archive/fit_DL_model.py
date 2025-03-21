#!/usr/bin/python3    
# standard python packages
import os, sys
from os.path import dirname, abspath, join
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import random
import math
from joblib import Parallel, delayed  
from copy import copy, deepcopy
from tqdm import tqdm
import argparse
from datetime import datetime
import re


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchmetrics
from torchvision import transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


# import toybrains library
# allows users to run this script from anywhere
TOYBRAINS_DIR = abspath(join(dirname(__file__), '../../../toybrains/'))
if TOYBRAINS_DIR not in sys.path: sys.path.append(TOYBRAINS_DIR)
from utils.DLutils import *
from utils.multiprocess import *

# disable some unneccesary lightning warnings
import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

# set GPU settings
torch.set_float32_matmul_precision('medium')
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

DEEPREPVIZ_REPO = abspath(join(dirname(__file__), "../../../Deep-confound-control-v2/")) 
# check that DeepRepViz repo is available and import it
assert os.path.isdir(DEEPREPVIZ_REPO) and os.path.exists(DEEPREPVIZ_REPO+'/DeepRepViz.py'), f"No DeepRepViz repository found in {DEEPREPVIZ_REPO}. Download the repo from https://github.com/ritterlab/Deep-confound-control-v2 and add its relative path to 'DEEPREPVIZ_REPO'."
if DEEPREPVIZ_REPO not in sys.path: sys.path.append(DEEPREPVIZ_REPO)
from DeepRepViz import DeepRepViz


DEEPREPVIZ_BACKEND = abspath(join(DEEPREPVIZ_REPO, "application/backend/deep_confound_control/core/")) 
assert os.path.isdir(DEEPREPVIZ_BACKEND) and os.path.exists(DEEPREPVIZ_BACKEND+'/DeepRepVizBackend.py'), f"No DeepRepViz repository found in {DEEPREPVIZ_BACKEND}. Add the correct relative path to the backend to the 'DEEPREPVIZ_BACKEND' global variable."
if DEEPREPVIZ_BACKEND not in sys.path:
    sys.path.append(DEEPREPVIZ_BACKEND)
from DeepRepVizBackend import DeepRepVizBackend


###############################################################################
####################       LIGHTNING trainer    ###############################
###############################################################################


class LightningModel(L.LightningModule):
    
    def __init__(self, model, learning_rate,
                 task="binary", num_classes=1):
        '''LightningModule that receives a PyTorch model as input'''
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.num_classes = num_classes
        # self.metric_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes) 
        self._metric_spec = torchmetrics.Specificity(task=task, num_classes=num_classes)
        self._metric_recall = torchmetrics.Recall(task=task, num_classes=num_classes)
        self.metric_D2 = D2metric() 
        self.metric_deviance = DevianceMetric()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        # compute all metrics on the predictions
        if self.num_classes==1:
            logits = torch.squeeze(logits, dim=-1)
            true_labels = true_labels.to(torch.float32)
            # print(logits.shape, true_labels.shape)
            loss = F.binary_cross_entropy_with_logits(logits, true_labels)
            predicted_labels = torch.sigmoid(logits)>0.5
            
        else:
            loss = F.cross_entropy(logits, true_labels)
            predicted_labels = torch.argmax(logits, dim=1)
        # acc = self.metric_acc(predicted_labels, true_labels)
        # calculate balanced accuracy
        spec = self._metric_spec(predicted_labels, true_labels)
        recall = self._metric_recall(predicted_labels, true_labels)
        BAC = (spec+recall)/2
        D2 = self.metric_D2(logits, true_labels)
        Deviance = self.metric_deviance(logits, true_labels)
        metrics = {'loss':loss, 'BAC':BAC, 'D2':D2}
        return true_labels, logits, metrics

    def training_step(self, batch, batch_idx):
        labels, preds, metrics = self._shared_step(batch)
        # append 'train_' to every key
        log_metrics = {'train_'+k:v for k,v in metrics.items()}
        self.log_dict(log_metrics,
                      prog_bar=True, 
                      on_epoch=True, on_step=False)
        return log_metrics['train_loss']

    def validation_step(self, batch, batch_idx):
        labels, preds, metrics = self._shared_step(batch)
        # append 'val_' to every key
        log_metrics = {'val_'+k:v for k,v in metrics.items()}
        self.log_dict(log_metrics,
                      prog_bar=True, 
                      on_epoch=True, on_step=False)
        return labels, preds

    def test_step(self, batch, batch_idx):
        labels, preds, metrics = self._shared_step(batch)
        # append 'val_' to every key
        log_metrics = {'test_'+k:v for k,v in metrics.items()}
        self.log_dict(log_metrics,
                      prog_bar=True, 
                      on_epoch=True, on_step=False)
        return labels, preds
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # DeepRepViz returns ids and the normal batch outputs as tuples
        ids, batch = batch
        true_labels, logits, metrics = self._shared_step(batch)
        return ids, true_labels, logits, metrics

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                         factor=0.75, patience=3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch", # default
                "frequency": 1, # default
            },
        }
    
###########################################################################################
############       MAIN function: train deep learning models on toybrains     #############
###########################################################################################

def fit_DL_model(dataset_path, label_col, 
                datasplit_df, trial='trial_0',
                ID_col='subjectID', 
                additional_drv_test_data={},
                model_class=SimpleCNN,
                learning_rate=0.05,
                model_kwargs=dict(num_classes=2, final_act_size=65),
                trainer_args=dict(max_epochs=50, accelerator='gpu',
                                    devices=[1]),
                additional_loggers=[],
                additional_callbacks = [],
                batch_size=64, num_workers=8,
                early_stop_patience=8,
                gen_v1_table=False,
                show_batch=False, random_seed=None, debug=False,
                unique_name=''):

    # forcefully set a random seed in debug mode
    if random_seed is None and debug:
        random_seed=42
    if random_seed is not None:
        torch.manual_seed(random_seed) 
        np.random.seed(random_seed)
        random.seed(random_seed)
        L.seed_everything(random_seed)
    
    # load the dataset
    dataset_name = os.path.basename(dataset_path)

    # split the dataset as defined in the datasplit_df
    if datasplit_df.index.name==ID_col: datasplit_df = datasplit_df.reset_index()
    # Select a specific trial given by 'trial' out of the k-folds  
    split_col = "datasplit"
    datasplit_df = datasplit_df.rename(columns={trial:split_col})
    datasplit_df = datasplit_df[[ID_col, label_col, split_col]]
    
    df_train = datasplit_df[datasplit_df[split_col]=='train']    
    df_val = datasplit_df[datasplit_df[split_col]=='val']

    print(f"Dataset: {dataset_name} \n\tTraining data split = {len(df_train)} \
\n\tValidation data split = {len(df_val)} \n\tTest datasets = {list(additional_drv_test_data.keys())}")
    
    # create pytorch data loaders
    train_dataset = ToyBrainsDataloader(
        img_names = df_train[ID_col].values, # TODO change hardcoded
        labels = df_train[label_col].values,
        img_dir = dataset_path+'/train/images',
        transform = transforms.Compose([transforms.ToTensor()])
        )
    train_loader = DataLoader(
                    dataset=train_dataset,
                    shuffle=True, batch_size=batch_size, drop_last=True,
                    num_workers=num_workers, 
                    )
    
    val_dataset = ToyBrainsDataloader(
        img_names = df_val[ID_col].values, 
        labels = df_val[label_col].values,
        img_dir = dataset_path+'/train/images',
        transform = transforms.Compose([transforms.ToTensor()])
        )
    val_loader = DataLoader(
                    dataset=val_dataset,
                    shuffle=False, batch_size=batch_size, drop_last=True,
                    num_workers=num_workers, 
                    )
    
    if show_batch:
        viz_batch(val_loader, title="Validation data")
    
    # create dataloaders for DeepRepViz() with no shuffle
    drv_train_dataset = {
            'dataloader_kwargs': dict(img_dir=dataset_path+'/train/images',
                                    img_names=df_train[ID_col].values,
                                    labels=df_train[LABEL_COL].values,
                                    transform=transforms.ToTensor()),
            "expected_IDs":df_train[ID_col].values,  
            "expected_labels": LabelEncoder().fit_transform(df_train[LABEL_COL].values)}

    drv_test_datasets = {
        'val': {
            'dataloader_kwargs': dict(img_dir=dataset_path+'/train/images',
                                    img_names=df_val[ID_col].values,
                                    labels=df_val[LABEL_COL].values,
                                    transform=transforms.ToTensor()),
            "expected_IDs":df_val[ID_col].values,
            "expected_labels": LabelEncoder().fit_transform(df_val[LABEL_COL].values)}
        }    

    # append any additional test datasets provided too
    for testdata_name, testdata_path in additional_drv_test_data.items():
        # pass test datasets to DeepRepViz that will test the best model and log the results in best_checkpoint.json
        test_data =  glob(testdata_path + '/toybrains*.csv')
        assert len(test_data)==1, f"Multiple or no test data found in {testdata_path}. Found = {test_data}"
        df_test = pd.read_csv(test_data[0])
        drv_test_datasets[testdata_name] = {
            'dataloader_kwargs': dict(img_dir=testdata_path+'/images',
                                    img_names=df_test[ID_col].values,
                                    labels=df_test[label_col].values,
                                    transform=transforms.ToTensor()),
            "expected_IDs"   :df_test[ID_col].values,
            "expected_labels":LabelEncoder().fit_transform(df_test[label_col].values)}

    # load model
    model = model_class(**model_kwargs)
    lightning_model = LightningModel(model, learning_rate=learning_rate, 
                                     num_classes=model_kwargs['num_classes'])

    # configure TensorBoardLogger as the main logger 
    # create a unique name for the logs based on the dataset, model and user provided suffix
    if unique_name != '': unique_name = '_' + unique_name
    unique_name = f'{dataset_name}_{model_class.__name__}{unique_name}'
    logger = TensorBoardLogger(save_dir='log', name=unique_name, version=trial) 
    if additional_loggers: # plus, any additional user provided loggers
        logger = [logger] + additional_loggers
    
    ## Init DeepRepViz callback            
    drv = DeepRepViz(dataloader_class=ToyBrainsDataloader, 
                    dataset_kwargs=drv_train_dataset,
                    datasets_kwargs_test=drv_test_datasets,
                    hook_layer=-1,
                    best_ckpt_by='loss_val', best_ckpt_metric_should_be='min',
                    verbose=int(debug))
    
    callbacks = additional_callbacks + [drv]
    # add any other callbacks
    if early_stop_patience:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", 
                                       patience=early_stop_patience))
    
    # train model
    trainer = L.Trainer(callbacks=callbacks,
                        logger=logger,
                        overfit_batches = 5 if debug else 0,
                        log_every_n_steps= 2 if debug else 20,
                        **trainer_args) # deterministic=True
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)

    # create and save the DeepRepViz v1 table 
    if gen_v1_table:
        raw_csv_path = glob(f'{dataset_path}/*{dataset_name}.csv')[0]
        
        df_data = pd.read_csv(raw_csv_path)
        drv_backend = DeepRepVizBackend(
                    conf_table=df_data,
                    best_ckpt_by='loss_val',
                    ID_col=ID_col, label_col=label_col)
        
        log_dir = trainer.log_dir + '/deeprepvizlog/'
        drv_backend.load_log(log_dir)

        drv_backend.convert_log_to_v1_table(log_key=log_dir, unique_name=unique_name)
    
    return trainer, logger

###########################################################################################
########################             MAIN function end             ########################
###########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/toybrains_n10000', type=str,
                        help='The relative pathway of the generated dataset in the toybrains repo')
    parser.add_argument('-k', '--k_fold', default=1, type=int)
    parser.add_argument( '--no_ood_val', action='store_true')
    parser.add_argument('-e', '--max_epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('--gpus', nargs='+', default=None, type=int)
    parser.add_argument('--final_act_size', default=64, type=int)
    parser.add_argument('-n', '--unique_name', default='', type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    parser.add_argument('-r', '--random_seed', default=None, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float)
    # parser.add_argument('-j','--n_jobs', default=20, type=int)
    args = parser.parse_args()
    
    # next check that the toybrains dataset is generated and available
    DATA_DIR = os.path.abspath(args.data_dir)
    DATA_CSV = glob(DATA_DIR + '/train/toybrains*.csv')
    assert len(DATA_CSV)==1, f"Toybrains dataset found = {DATA_CSV}.\
 \nEnsure that that the dataset {args.data_dir} is generated using the `1_create_toybrains.py` script in the toybrains repo. \
Also ensure only one dataset exists for the given query '{DATA_DIR}'."
    DATA_CSV = DATA_CSV[0]
    ID_COL = 'subjectID'
    LABEL_COL = 'lbl_y'
    N_SAMPLES = int(DATA_DIR.split('_n')[-1].split('_')[0])

    unique_name = args.unique_name
    if args.debug:
        os.system('rm -rf log/*debugmode*')
        unique_name = 'debugmode'+unique_name

        args.max_epochs = 2 if args.max_epochs>=100 else args.max_epochs
        args.batch_size = 5
        args.k_fold = 2 if args.k_fold>2 else args.k_fold
        num_workers = 5
        # args.no_ood_val = False
    else:
        num_workers = 8

    start_time = datetime.now()
    
    # prepare the data splits as a dataframe mapping the subjectID to the split and trial
    data = pd.read_csv(DATA_CSV)
    assert ID_COL in data.columns, f"ID_COL={ID_COL} is not present in the dataset's csv file. \
Available colnames = {data.columns.tolist()}"
    assert LABEL_COL in data.columns, f"LABEL_COL={LABEL_COL} is not present in the dataset's csv file. \
Available colnames = {data.columns.tolist()}"

    ### SPLITS: Create the n-fold splits for the data
    # use only the columns subjectID and label to create the splits
    datasplit_df = data.drop(columns=[c for c in data.columns if c not in [ID_COL, LABEL_COL]])
    datasplit_df = datasplit_df.set_index(ID_COL)
    # create 'trial_x' columns: init as columns as args.k_fold
    for trial in range(args.k_fold): 
        datasplit_df[f'trial_{trial}'] = 'unknown'

    trainval_idxs = datasplit_df.index
    
    splitter = StratifiedShuffleSplit(n_splits=args.k_fold, test_size=0.1,
                                        random_state=args.random_seed)
    splits = splitter.split(trainval_idxs, y=datasplit_df[LABEL_COL])
    for trial_idx, (train_idxs_i, val_idxs_i) in enumerate(splits): 
        datasplit_df.loc[trainval_idxs[train_idxs_i], f'trial_{trial_idx}'] = 'train'
        datasplit_df.loc[trainval_idxs[val_idxs_i], f'trial_{trial_idx}']   = 'val'

    datasplit_df = datasplit_df.sort_index()
    # ensure that all data points are assigned to either train or val split
    assert (datasplit_df.filter(like='trial').map(lambda x: x!='unknown')).all().all(), "some data points are not assigned to any split. {}".format(datasplit_df)

    ### collect all test datasets available
    test_datasets = {os.path.basename(d):d for d in glob(DATA_DIR + '/test_*')}
    if args.no_ood_val: test_datasets = {k:v for k,v in test_datasets.items() if 'all' not in k}

    ### MODEL: configure the DL model
    DL_MODEL = SimpleCNN
    model_kwargs = dict(num_classes=1, final_act_size=args.final_act_size)
    
    ### TRAIN: parallize the training across trials
    def _run_one_trial(trial):
        # use whatever is available (CPU/GPU) if args.gpu is None  
        accelerator= "gpu" if args.gpus is not None else "auto"
        devices=args.gpus if args.gpus is not None else [1] #TODO use multiple GPUs? (args.gpus+trial)%torch.cuda.device_count()
        
        trainer, logger = fit_DL_model(
                                DATA_DIR, 
                                label_col=LABEL_COL, ID_col=ID_COL, 
                                trial=f'trial_{trial}', datasplit_df=datasplit_df, 
                                additional_drv_test_data=test_datasets,
                                model_class=DL_MODEL, model_kwargs=model_kwargs,
                                debug=args.debug, 
                                learning_rate=args.learning_rate,
                                additional_callbacks=[] , 
                                additional_loggers=[],
                                batch_size=args.batch_size, num_workers=num_workers,
                                trainer_args=dict(
                                    max_epochs=args.max_epochs, 
                                    accelerator=accelerator,
                                    devices=devices),
                                unique_name=unique_name)
        
    # run all trials in parallel by creating a pool of workers
    if args.k_fold <= 1:
        _run_one_trial(0)
    else:
        processes = []
        for trial in range(args.k_fold):
            p = mp.Process(target=_run_one_trial, args=(trial,))
            p.start()
            processes.append(p)
        # wait for all processes to finish
        for p in processes: p.join()

    # runtime
    total_time = datetime.now() - start_time
    minutes, seconds = divmod(total_time.total_seconds(), 60)
    print(f"Total runtime: {int(minutes)} minutes {int(seconds)} seconds")
    