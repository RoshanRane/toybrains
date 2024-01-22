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
import torch.multiprocessing as mp

from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision import transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import logging
# disable some unneccesary lightning warnings
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

# import toybrains library
# allows users to run this script from anywhere
TOYBRAINS_DIR = abspath(join(dirname(__file__), '../'))
if TOYBRAINS_DIR not in sys.path:
    sys.path.append(TOYBRAINS_DIR)
from utils.DLutils import *
from utils.multiprocess import *

# set GPU settings
torch.set_float32_matmul_precision('medium')
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

DEEPREPVIZ_REPO = abspath(join(dirname(__file__), "../../Deep-confound-control-v2/")) 
# check that DeepRepViz repo is available and import it
assert os.path.isdir(DEEPREPVIZ_REPO) and os.path.exists(DEEPREPVIZ_REPO+'/DeepRepViz.py'), f"No DeepRepViz repository found in {DEEPREPVIZ_REPO}. Download the repo from https://github.com/ritterlab/Deep-confound-control-v2 and add its relative path to 'DEEPREPVIZ_REPO'."
if DEEPREPVIZ_REPO not in sys.path:
    sys.path.append(DEEPREPVIZ_REPO)
from DeepRepViz import DeepRepViz
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
        
    def predict_step(self, batch, batch_idx):
        # DeepRepViz returns ids and the normal batch outputs as tuples
        ids, batch = batch
        true_labels, logits, metrics = self._shared_step(batch)
        return ids, true_labels, logits, metrics

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
                                                         factor=0.1, patience=3)
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
                model_class=SimpleCNN,
                model_kwargs=dict(num_classes=2, final_act_size=65),
                trainer_args=dict(max_epochs=50, accelerator='gpu',
                                    devices=[1]),
                additional_loggers=[],
                additional_callbacks = [],
                batch_size=64, num_workers=8,
                early_stop_patience=6,
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
    dataset_unique_name = dataset_path.split('/')[-1].split('_')[-1]

    # split the dataset as defined in the datasplit_df
    if datasplit_df.index.name==ID_col:
        datasplit_df = datasplit_df.reset_index()
    split_col = "datasplit"
    # select the correct trial
    datasplit_df = datasplit_df.rename(columns={trial:split_col})
    datasplit_df = datasplit_df[[ID_col, label_col, split_col]]
    
    # datasplit_df = datasplit_df.rename(columns={label_col:'label'}) #TODO remove this hardcoded requirement of label_col name from get_toybrain_dataloader()
    df_train = datasplit_df[datasplit_df[split_col]=='train']    
    df_val = datasplit_df[datasplit_df[split_col]=='val']
    df_test = datasplit_df[datasplit_df[split_col]=='test']

    print(f"Dataset: {dataset_path} ({dataset_unique_name})\n  Training data split = {len(df_train)} \n\
  Validation data split = {len(df_val)} \n  Test data split = {len(df_test)}")
    
    # create pytorch data loaders
    train_dataset = ToyBrainsDataloader(
        img_names = df_train[ID_col].values, # TODO change hardcoded
        labels = df_train[label_col].values,
        img_dir = dataset_path+'/images',
        transform = transforms.Compose([transforms.ToTensor()])
        )
    train_loader = DataLoader(
                    dataset=train_dataset,
                    shuffle=True, batch_size=batch_size, drop_last=True,
                    num_workers=num_workers, 
                    )
    
    val_dataset = ToyBrainsDataloader(
        img_names = df_val[ID_col].values, # TODO change hardcoded
        labels = df_val[label_col].values,
        img_dir = dataset_path+'/images',
        transform = transforms.Compose([transforms.ToTensor()])
        )
    val_loader = DataLoader(
                    dataset=val_dataset,
                    shuffle=False, batch_size=batch_size, drop_last=True,
                    num_workers=num_workers, 
                    )
    
    df_test = datasplit_df[datasplit_df[split_col]=='test']
    test_dataset = ToyBrainsDataloader(
        img_names = df_test[ID_col].values, # TODO change hardcoded
        labels = df_test[label_col].values,
        img_dir = dataset_path+'/images',
        transform = transforms.Compose([transforms.ToTensor()])
        )
    test_loader = DataLoader(
                    dataset=test_dataset,
                    shuffle=False, batch_size=batch_size, drop_last=True,
                    num_workers=num_workers, 
                    )
    
    if show_batch:
        viz_batch(val_loader, title="Validation data")
    
    # create a dataloader for DeepRepViz with the whole data and no shuffle
    # collect the values for deeprepviz
    IDs = datasplit_df[ID_col].values
    expected_labels = datasplit_df[LABEL_COL].values
    datasplits = datasplit_df[split_col].values

    drv_loader_kwargs = dict(
                    img_dir=dataset_path+'/images',
                    img_names=IDs,
                    labels=expected_labels,
                    transform=transforms.ToTensor())
    
    deeprepviz_kwargs = dict(
                    dataloader_class=ToyBrainsDataloader, 
                    dataloader_kwargs=drv_loader_kwargs,
                    expected_IDs=IDs, 
                    expected_labels=expected_labels, 
                    datasplits=datasplits,
                    hook_layer=-1,
                    debug=False)

    # load model
    model = model_class(**model_kwargs)
    lightning_model = LightningModel(model, learning_rate=0.05, 
                                     num_classes=model_kwargs['num_classes'])

    # configure TensorBoardLogger as the main logger 
    # create a unique name for the logs based on the dataset, model and user provided suffix
    unique_name = f'toybrains-{dataset_unique_name}_{model_class.__name__}' + unique_name
    logger = TensorBoardLogger(save_dir='log', name=unique_name, version=trial) 
    if additional_loggers: # plus, any additional user provided loggers
        logger = [logger] + additional_loggers
    
    ## Init DeepRepViz callback            
    drv = DeepRepViz(**deeprepviz_kwargs)
    callbacks = additional_callbacks + [drv]
    # add any other callbacks
    if early_stop_patience:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", 
                                       patience=early_stop_patience))
    
    # train model
    trainer = L.Trainer(callbacks=callbacks,
                        logger=logger,
                        overfit_batches = 5 if debug else 0,
                        log_every_n_steps= 2 if debug else 50,
                        **trainer_args) # deterministic=True
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)
        
    # test model
    test_scores = trainer.test(lightning_model, verbose=False,
                               dataloaders=test_loader,
                               ckpt_path="best")[0]

    print("Test data performance with the best model:\n\
-------------------------------------------------------\n\
Dataset      = {} ({})\n\
Balanced Acc = {:.2f}% \t D2 = {:.2f}%".format(
        dataset_path, dataset_unique_name, 
         test_scores['test_BAC']*100,  test_scores['test_D2']*100))
    
    # create and save the DeepRepViz v1 table 
    raw_csv_path = glob(f'{dataset_path}/*{dataset_unique_name}.csv')[0]
    df_data = pd.read_csv(raw_csv_path)
    drv_backend = DeepRepVizBackend(
                  conf_table=df_data,
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
    parser.add_argument('-e', '--max_epochs', default=50, type=int)
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('--gpus', nargs='+', default=None, type=int)
    parser.add_argument('--final_act_size', default=64, type=int)
    parser.add_argument('-n', '--unique_name', default='', type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    parser.add_argument('-r', '--random_seed', default=None, type=int)
    parser.add_argument('-k', '--k_fold', default=1, type=int)
    # parser.add_argument('-j','--n_jobs', default=20, type=int)
    args = parser.parse_args()
    
    # next check that the toybrains dataset is generated and available
    DATA_DIR = os.path.join(TOYBRAINS_DIR, args.data_dir)
    DATA_CSV = glob(DATA_DIR + '/toybrains*.csv')
    assert len(DATA_CSV)==1, f"No toybrains dataset was found in {DATA_DIR}. Ensure that that the dataset {args.data_dir} is generated using the `create_toybrains.py` script in the toybrains repo. Also cross check that the dataset directory path you have provided here = '{args.data_dir}'  is correct."
    DATA_CSV = DATA_CSV[0]
    ID_COL = 'subjectID'
    LABEL_COL = 'lbl_lesion'
    
    unique_name = 'debugmode' if args.debug else args.unique_name
    if args.debug:
        args.max_epochs = 1
        args.batch_size = 5
        # args.k_fold = 1
        num_workers = 5
    else:
        num_workers = 8

    start_time = datetime.now()

    # prepare the data splits as a dataframe mapping the subjectID to the split and trial
    data = pd.read_csv(DATA_CSV)
    assert ID_COL in data.columns, f"ID_COL={ID_COL} is not present in the dataset's csv file. \
Available colnames = {data.columns.tolist()}"
    assert LABEL_COL in data.columns, f"LABEL_COL={LABEL_COL} is not present in the dataset's csv file. \
Available colnames = {data.columns.tolist()}"
    # drop all columns except subjectID, label, trial and datasplit
    datasplit_df = data.drop(columns=[c for c in data.columns if c not in [ID_COL, LABEL_COL]])
    datasplit_df = datasplit_df.set_index(ID_COL)
    # init as many trial columns as requested in args.k_fold
    for trial in range(args.k_fold):
        datasplit_df[f'trial_{trial}'] = 'unknown'
    # first, set aside 20% of the data as test
    train_idxs, test_idxs = train_test_split(datasplit_df.index, test_size=0.2,
                                             random_state=args.random_seed)
    for trial in range(args.k_fold):
        datasplit_df.loc[test_idxs, f'trial_{trial}'] = 'test'
    # initialize such that all data is used in the first trial
    if args.k_fold <= 1:
        train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2, 
                                               random_state=args.random_seed)
        datasplit_df.loc[train_idxs, 'trial_0'] = 'train'
        datasplit_df.loc[val_idxs, 'trial_0'] = 'val'
    else:
        splitter = StratifiedKFold(n_splits=args.k_fold,
                                   shuffle=True,
                                   random_state=args.random_seed)
        splits = splitter.split(train_idxs, y=datasplit_df.loc[train_idxs, LABEL_COL])
        for trial_idx, (train_idxs_i, val_idxs_i) in enumerate(splits): 
            datasplit_df.loc[train_idxs[train_idxs_i], f'trial_{trial_idx}'] = 'train'
            datasplit_df.loc[train_idxs[val_idxs_i], f'trial_{trial_idx}'] = 'val'

    datasplit_df = datasplit_df.sort_index()
    (datasplit_df.filter(like='trial')!='unknown').all(), "some data points are not assigned to any split. {}".format(datasplit_df)

    # configure the DL model
    DL_MODEL = SimpleCNN
    model_kwargs = dict(num_classes=1, final_act_size=args.final_act_size)
    
    def _run_one_trial(trial):
        # use whatever is available (CPU/GPU) if args.gpu is None  
        accelerator= "gpu" if args.gpus is not None else "auto"
        devices=args.gpus if args.gpus is not None else [1] #TODO use multiple GPUs? (args.gpus+trial)%torch.cuda.device_count()
        
        trainer, logger = fit_DL_model(
                                DATA_DIR, 
                                label_col=LABEL_COL, ID_col=ID_COL, 
                                datasplit_df=datasplit_df.reset_index(), trial=f'trial_{trial}',
                                model_class=DL_MODEL, model_kwargs=model_kwargs,
                                debug=args.debug, 
                                additional_callbacks=[],
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