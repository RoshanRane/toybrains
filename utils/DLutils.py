import os, sys
from glob import glob
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# deep learning imports
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
import torchvision, torchmetrics
from torchvision import datasets, transforms
import lightning as L

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# import monai

# custom imports 
from utils.dataset import split_dataset
from utils.metrics import explained_deviance



###########################################################################################
############################          Dataloader         ##################################
###########################################################################################

class ToyBrainsDataloader(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # based on Dataframe columns
        self.img_names = df["subjectID"]
        self.labels = df["label"]
        
    def __getitem__(self, index):
        
        number = str(self.img_names[index]).zfill(5)
        img = Image.open(os.path.join(self.img_dir, number + ".jpg"))
        
        if self.transform is not None:
            img = self.transform(img)
            
        label = torch.as_tensor(self.labels[index]).type(torch.LongTensor)
        return img, label
    
    def __len__(self):
        return self.labels.shape[0]
    
    
    
def get_dataset_loaders(data_split_dfs,
                        data_dir="toybrains", 
                        batch_size=16, 
                        shuffles=[True, False, False], 
                        num_workers=30, transform=[],
                        ):
    ''' Creates pytorch dataloaders of the ToyBrainsDataloader for all the 
    dataframes passed in *args

    NOTES
    ----
    reference: https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit05-lightning/5.5-datamodules
    TODO: refactoring ToyBrainsDataModule
    '''
    data_loaders = []
    assert len(shuffles) == len(data_split_dfs)
    
    for i,data_split_df in enumerate(data_split_dfs):
        dataset = ToyBrainsDataloader(
                    df=data_split_df,
                    img_dir=f'{data_dir}/images',
                    transform=transforms.Compose(
                        [transforms.ToTensor()]+transform)
                    )
        data_loader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffles[i],
                        num_workers=num_workers
                    )
        data_loaders.append(data_loader)

    return data_loaders
    

#################################################################################################
####################       LIGHTNING trainer class and helpers    ###############################
#################################################################################################


class LightningModel(L.LightningModule):
    
    def __init__(self, model, learning_rate,
                 task="binary", num_classes=2):
        '''LightningModule that receives a PyTorch model as input'''
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
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
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        # acc = self.metric_acc(predicted_labels, true_labels)
        # calculate balanced accuracy
        spec = self._metric_spec(predicted_labels, true_labels)
        recall = self._metric_recall(predicted_labels, true_labels)
        BAC = (spec+recall)/2
        D2 = self.metric_D2(logits, true_labels)
        return {'loss':loss, 'BAC':BAC, 'D2':D2}

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        # append 'train_' to every key
        metrics = {'train_'+k:v for k,v in metrics.items()}
        self.log_dict(metrics,
                      prog_bar=True, 
                      on_epoch=True, on_step=False)
        return metrics['train_loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        # append 'val_' to every key
        metrics = {'val_'+k:v for k,v in metrics.items()}
        self.log_dict(metrics,
                      prog_bar=True, 
                      on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        # append 'val_' to every key
        metrics = {'test_'+k:v for k,v in metrics.items()}
        self.log_dict(metrics,
                      prog_bar=True, 
                      on_epoch=True, on_step=False)

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


class D2metric(Metric):
    def __init__(self, num_classes=2):
        super().__init__()
        self.add_state("targets", default=torch.Tensor([]))
        self.add_state("logits",  default=torch.Tensor([]))
        self.unique_y = list(range(num_classes))

    def update(self, logit: torch.Tensor, target: torch.Tensor):
        assert len(target) == len(logit), f"target.shape={target.shape} but logits.shape={logit.shape}"
        # if setting for the first time
        if len(self.targets)==0:
            self.targets = target
            self.logits = logit
        else:
            self.targets = torch.cat([self.targets,target], dim=0) 
            self.logits  = torch.cat([self.logits, logit ], dim=0)
            

    def compute(self):
        return explained_deviance(
            self.targets.detach().cpu(), 
            y_pred_logits=self.logits.detach().cpu(), 
            unique_y=self.unique_y)
    
#################################################################################################
# PyTorch Model
#################################################################################################

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, num_features, num_classes, bias):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes, bias)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.linear(x)
        return logits

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # convolutional layers
        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=16),
            nn.Dropout(0.1),
            ConvBlock(in_channels=16, out_channels=32),
            nn.Dropout(0.1),
            ConvBlock(in_channels=32, out_channels=64),
        )
        
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            # TODO hardcoded input size
            nn.Linear(64 * 8 * 8, 3),
            nn.Linear(3, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

#################################################################################################
# Visualization
#################################################################################################

# function

def viz_batch(loader, title="Training images", debug=False):
    '''
    Visualize the batch
    
    Reference
    ---------
    https://github.com/Lightning-AI/dl-fundamentals/blob/main/unit04-multilayer-nets/4.3-mlp-pytorch/4.3-mlp-pytorch-part3-5-mnist/4.3-mlp-pytorch-part5-mnist.ipynb
    '''
    imgs, lbls = [], []
    for batch_imgs, batch_lbls in loader:
        imgs.extend(batch_imgs)
        lbls.extend(batch_lbls.numpy().astype(int).tolist())
        if len(imgs)>=4: break
            
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    lbls_str = str(lbls)
    if len(lbls_str)>100:
        lbls_str = lbls_str[:100]+'\n'+lbls_str[100:]
    plt.title(title+f"\nlabels={lbls_str}")
    plt.imshow(
        np.transpose(
        torchvision.utils.make_grid(
            imgs, padding=1, pad_value=1.0, normalize=True), 
        (1, 2, 0)))
    plt.show()
    
    
    
###########################################################################################
############       MAIN function: train deep learning models on toybrains     #############
###########################################################################################

def fit_DL_model(dataset_path,
                label,
                model = SimpleCNN(num_classes=2),
                logger = CSVLogger(save_dir='logs'),
                batch_size=64,
                show_batch=False, show_training_curves=True,
                early_stop_patience=6,
                random_seed=None, debug=False, 
                trainer_args={"max_epochs":10, 
                              "accelerator":'gpu',
                              "devices":[1]}):
    
    if debug:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('-'*40+f"\nTraining Deep learning model on toybrains:\n\
        dataset_path: {dataset_path} \n\
        model: {model.__class__.__name__}(n_params={trainable_params})\n\
        training_args: {trainer_args}\n"+'-'*40)
        os.environ["CUDA_LAUNCH_BLOCKING"]="1"
        random_seed = 42
        trainer_args["max_epochs"]=2 if trainer_args["max_epochs"]>2 else trainer_args["max_epochs"]

    # set GPU settings
    torch.set_float32_matmul_precision('medium')
    # forcefully set a random seed in debug mode
    if random_seed is None and debug:
        random_seed=42
    if random_seed is not None:
        torch.manual_seed(random_seed) 
        np.random.seed(random_seed)
        random.seed(random_seed)
        # set the seed for Lightning
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
        if len(df_test)>500: df_test = df_val.iloc[:500]
    
    print(f"Dataset: {dataset_path} ({unique_name})\n  Training data split = {len(df_train)} \n \
  Validation data split = {len(df_val)} \n  Test data split = {len(df_test)}")
    
    # generate data loaders
    train_loader, val_loader, test_loader = get_dataset_loaders(
                    data_split_dfs=[df_train, df_val, df_test],
                    shuffles=[True,False,False],
                    data_dir=dataset_path,
                    batch_size=batch_size, 
                    num_workers=20, transform=[])
    if show_batch or debug:
        viz_batch(val_loader, title="Validation data")

    # load model
    lightning_model = LightningModel(model, learning_rate=0.05)
    # configure trainer settings
    logger.__setattr__("_name" , unique_name)
    callbacks = [ModelCheckpoint(dirpath=logger.log_dir,
                                 monitor="val_loss", mode="min",  
                                 save_top_k=1, save_last=True)]
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
    if show_training_curves:
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
