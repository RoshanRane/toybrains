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
from tqdm.auto import tqdm
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision, torchmetrics
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

def fit_DL_model(dataset_path,
                label,
                model_class=SimpleCNN,
                model_kwargs=dict(num_classes=2, final_act_size=65),
                trainer_args=dict(max_epochs=50, accelerator='gpu',
                                  devices=[1]),
                additional_loggers=[],
                additional_callbacks = [],
                batch_size=4, num_workers=0,
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
    raw_csv_path = glob(f'{dataset_path}/*{dataset_unique_name}.csv')[0]
    df_data = pd.read_csv(raw_csv_path)
    # split the dataset
    df_train, df_val, df_test = split_dataset(df_data, label, random_seed)
    print(f"Dataset: {dataset_path} ({dataset_unique_name})\n  Training data split = {len(df_train)} \n \
  Validation data split = {len(df_val)} \n  Test data split = {len(df_test)}")
    
    # generate data loaders
    common_settings = dict(images_dir=dataset_path+'/images',
                           batch_size=batch_size, 
                           num_workers=num_workers)
    train_loader = get_toybrain_dataloader(df_train,
                                           **common_settings)
    val_loader = get_toybrain_dataloader(df_val, shuffle=False,
                                        **common_settings)
    test_loader = get_toybrain_dataloader(df_test, shuffle=False,
                                          **common_settings)
    
    if show_batch:
        viz_batch(val_loader, title="Validation data")
    
    
    # create a dataloader for DeepRepViz with the whole data and no shuffle
    split_colname = 'datasplit'
    ID_col = 'subjectID'
    # add the split info too
    df_train[split_colname] = 'train'
    df_val[split_colname]   = 'val'
    df_test[split_colname]  = 'test'
    df_data = pd.concat([df_train, df_val, df_test])
    IDs = df_data[ID_col].values
    expected_labels = df_data[label].values
    datasplits = df_data[split_colname].values

    drv_loader_kwargs = dict(
                    img_dir=dataset_path+'/images',
                    img_names=IDs,
                    labels=expected_labels,
                    transform=transforms.ToTensor())
    
    deeprepviz_kwargs = dict(
                    dataloader_class=ToyBrainsDataloader, 
                    dataloader_kwargs=drv_loader_kwargs,
                    expected_IDs=IDs, expected_labels=expected_labels, datasplits=datasplits,
                    hook_layer=-1,
                    debug=False)

    # load model
    model = model_class(**model_kwargs)
    lightning_model = LightningModel(model, learning_rate=0.05, 
                                     num_classes=model_kwargs['num_classes'])

    # configure TensorBoardLogger as the main logger 
    # create a unique name for the logs based on the dataset, model and user provided suffix
    unique_name = f'toybrains-{dataset_unique_name}_{model_class.__name__}' + unique_name
    logger = TensorBoardLogger(save_dir='log', name=unique_name) 
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
    drv_backend = DeepRepVizBackend(
                  conf_table=df_data,
                  ID_col=ID_col, label_col=label)
    log_dir = trainer.log_dir + '/deeprepvizlog/'
    drv_backend.load_log(log_dir)
    drv_backend.convert_log_to_v1_table(log_key=log_dir, unique_name=unique_name)
    
    return trainer, logger

###########################################################################################
########################             MAIN function end             ########################
###########################################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/toybrains_n10000_highsignal', type=str,
                        help='The relative pathway of the generated dataset in the toybrains repo')
    parser.add_argument('-e', '--max_epochs', default=50, type=int)
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('--gpus', nargs='+', default=None, type=int)
    parser.add_argument('--final_act_size', default=64, type=int)
    parser.add_argument('-n', '--unique_name', default='', type=str)
    parser.add_argument('-d', '--debug',  action='store_true')
    # parser.add_argument('-j','--n_jobs', default=20, type=int)
    args = parser.parse_args()
    
    # next check that the toybrains dataset is generated and available
    DATA_DIR = os.path.join(TOYBRAINS_DIR, args.data_dir)
    DATA_CSV = glob(DATA_DIR + '/toybrains*.csv')
    assert len(DATA_CSV)==1, f"No toybrains dataset was found in {DATA_DIR}. Ensure that that the dataset {args.data_dir} is generated using the `create_toybrains.py` script in the toybrains repo. Also cross check that the dataset directory path you have provided here = '{args.data_dir}'  is correct."
    DATA_CSV = DATA_CSV[0]
    # use whatever is available (CPU/GPU) if args.gpu is None  
    accelerator= "gpu" if args.gpus is not None else "auto"
    devices=args.gpus if args.gpus is not None else 1
        
    DL_MODEL = SimpleCNN
    model_kwargs = dict(num_classes=1, final_act_size=args.final_act_size)
    
    unique_name = 'debug-mode' if args.debug else args.unique_name
    max_epochs = 3 if args.debug else args.max_epochs
    num_workers = 0 if args.debug else os.cpu_count()
    batch_size = args.batch_size
    start_time = datetime.now()
    
    trainer, logger = fit_DL_model(
                            DATA_DIR,
                            label='lbl_lesion',
                            model_class=DL_MODEL, model_kwargs=model_kwargs,
                            debug=args.debug, 
                            additional_callbacks=[],
                            additional_loggers=[],
                            batch_size=batch_size, num_workers=num_workers,
                            trainer_args=dict(
                                max_epochs=max_epochs, 
                                accelerator=accelerator,
                                devices=devices),
                            unique_name=unique_name)
    
    # runtime
    total_time = datetime.now() - start_time
    minutes, seconds = divmod(total_time.total_seconds(), 60)
    print(f"Total runtime: {int(minutes)} minutes {int(seconds)} seconds")