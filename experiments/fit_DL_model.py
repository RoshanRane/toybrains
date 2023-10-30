#!/usr/bin/python3    
# standard python packages
# @arjun to run on CPU in dev mode do: $python3 toybrains_fit_DL.py -d -e 2 -b 4 
import os, sys
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

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
import torchvision, torchmetrics
from torchvision import datasets, transforms
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# import toybrains library
# check that Toybrains repo is available
TOYBRAINS_DIR = '../'
assert os.path.isdir(TOYBRAINS_DIR) and os.path.exists(TOYBRAINS_DIR+'/create_toybrains.py'), f"No toybrains repository found in {TOYBRAINS_DIR}. Download the toybrains dataset from https://github.com/RoshanRane/toybrains and save it at the relative directory path provided here with the --dir arg."
sys.path.append(TOYBRAINS_DIR)
from create_toybrains import ToyBrainsData
from utils.DLutils import *

# DEEPREPVIZ_REPO = "../../Deep-confound-control-v2/"
# sys.path.append(DEEPREPVIZ_REPO)
# from DeepRepVizLogger import DeepRepVizLogger


###############################################################################
####################       LIGHTNING trainer    ###############################
###############################################################################


class LightningModel(L.LightningModule):
    
    def __init__(self, model, learning_rate,
                 task="binary", num_classes=2):
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
                model=SimpleCNN,
                num_classes=2,
                logger = CSVLogger,
                logger_args = {'save_dir':'logs'},
                callbacks = [],
                batch_size=64,
                show_batch=False, show_training_curves=True,
                early_stop_patience=6,
                random_seed=None, debug=False, 
                trainer_args={"max_epochs":10, 
                              "accelerator":'gpu',
                              "devices":[1]}):

    # set GPU settings
    torch.set_float32_matmul_precision('medium')
    if debug:
        os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    # forcefully set a random seed in debug mode
    if random_seed is None and debug:
        random_seed=42
    if random_seed is not None:
        torch.manual_seed(random_seed) 
        np.random.seed(random_seed)
        random.seed(random_seed)
        L.seed_everything(random_seed)
    
    # load the dataset
    unique_name = dataset_path.split('/')[-1].split('_')[-1]
    raw_csv_path = glob(f'{dataset_path}/*{unique_name}.csv')[0]
    data_df = pd.read_csv(raw_csv_path)
    # split the dataset
    df_train, df_val, df_test = split_dataset(raw_csv_path, label, random_seed)
    # in debug mode reduce the datasize of training data to maximum 5000 samples
    if debug:
        if len(df_train)>5000: df_train = df_train.iloc[:5000]
        if len(df_val)>500: df_val = df_val.iloc[:500]
        if len(df_test)>500: df_test = df_test.iloc[:500]
    
    print(f"Dataset: {dataset_path} ({unique_name})\n  Training data split = {len(df_train)} \n \
  Validation data split = {len(df_val)} \n  Test data split = {len(df_test)}")
    
    # generate data loaders
    train_loader = get_toybrain_dataloader(
                    df_train,
                    images_path=dataset_path+'/images',
                    batch_size=batch_size)
    val_loader = get_toybrain_dataloader(
                    df_val,
                    images_path=dataset_path+'/images',
                    shuffle=False,
                    batch_size=batch_size)
    test_loader = get_toybrain_dataloader(
                    df_test,
                    images_path=dataset_path+'/images',
                    shuffle=False,
                    batch_size=batch_size)
    
    if show_batch or debug:
        viz_batch(val_loader, title="Validation data")

    # load model
    model = model(num_classes=num_classes)
    lightning_model = LightningModel(model, learning_rate=0.05, 
                                     num_classes=num_classes)
    # configure trainer settings
    logger = logger(version=unique_name, **logger_args)
    # callbacks.append(ModelCheckpoint(dirpath=logger.log_dir,
    #                          monitor="val_loss", mode="min",  
    #                          save_top_k=1, save_last=True))
    if early_stop_patience and not debug:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", 
                                       patience=early_stop_patience))
    
    # train model
    trainer = L.Trainer(callbacks=callbacks,
                        logger=logger,
                        **trainer_args) # deterministic=True
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)
    
    # show training curves
    if show_training_curves and not isinstance(logger, WandbLogger):
        metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
        aggreg_metrics = []
        agg_col = "epoch"
        for i, dfg in metrics.groupby(agg_col):
            agg = dict(dfg.mean())
            agg[agg_col] = int(i)
            aggreg_metrics.append(agg)
        
        f, axes = plt.subplots(1,2, sharex=True, 
                               constrained_layout=True, 
                               figsize=(7,3))
        df_metrics = pd.DataFrame(aggreg_metrics)
        df_metrics[["train_loss", "val_loss"]].plot(
            ylabel="Loss", ax=axes[0],
            grid=True, legend=True, xlabel="Epoch", 
        )
        df_metrics[["train_D2", "val_D2"]].plot(
            ylabel=r"$D^2$", ax=axes[1],
            grid=True, legend=True, xlabel="Epoch", ylim=(0,1)
        )
        plt.show()
        
    # test model
    test_scores = trainer.test(lightning_model, verbose=False,
                               dataloaders=test_loader,
                              ckpt_path="best")[0]

    print("Test data performance with the best model:\n\
-------------------------------------------------------\n\
Dataset      = {} ({})\n\
Balanced Acc = {:.2f}% \t D2 = {:.2f}%".format(
        dataset_path, unique_name, 
         test_scores['test_BAC']*100,  test_scores['test_D2']*100))
    
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
    devices=args.gpus if args.gpus is not None else [2]
        
    DL_MODEL = SimpleCNN(num_classes=2)
    logger = WandbLogger#DeepRepVizLogger(save_dir='./logs/')
    logger_args = dict(save_dir='log_wandb', project='toybrains', log_model="all")
    callbacks = []
    
    trainer, logger = fit_DL_model(
                            DATA_DIR,
                            label='lbl_lesion',
                            model=DL_MODEL,
                            debug=args.debug, 
                            callbacks=callbacks,
                            logger=WandbLogger,
                            logger_args=logger_args,
                            show_training_curves=False,
                            trainer_args=dict(
                                max_epochs=args.max_epochs, 
                                accelerator=accelerator,
                                devices=devices))