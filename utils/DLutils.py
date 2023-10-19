import os, sys
from glob import glob
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# deep learning imports
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision, torchmetrics
import lightning as L
from torchvision import datasets, transforms
from lightning.pytorch.loggers import CSVLogger
# import monai

# custom imports 
from utils.dataset import split_dataset

#################################################################################################
# DATA MODULE
#################################################################################################
    

def get_dataset_loaders(data_split_dfs,
                        data_dir="toybrains", 
                        batch_size=16, 
                        shuffles=[True, False, False], 
                        num_workers=10, transform=[],
                        ):
    ''' Creates pytorch dataloaders of the ToyBrainsDataset for all the 
    dataframes passed in *args
    
    NOTES
    ----
    reference: https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit05-lightning/5.5-datamodules
    TODO: refactoring ToyBrainsDataModule
    '''
    # (TODO) Validation and Test should be set shuffle=False (wrong imlementation)
    data_loaders = []
    assert len(shuffles) == len(data_split_dfs)
    
    for i,data_split_df in enumerate(data_split_dfs):
        dataset = ToyBrainsDataset(
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

# classes

class ToyBrainsDataset(Dataset):
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

# LightningModule that receives a PyTorch model as input
# (TODO) add lr scheduler, etc.
# https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit06-dl-tips/6.2-learning-rates
class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.train_acc = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("accuracy", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
    
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
        probas = torch.sigmoid(logits)
        return probas

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
        probas = torch.sigmoid(logits)
        return probas

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
            ConvBlock(in_channels=3, out_channels=32),
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
        )
        
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            # TODO hardcoded input size
            nn.Linear(128 * 8 * 8, 128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        probas = torch.sigmoid(x)
        return probas

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
############       Main function: Deep learning models fit on toybrains       #############
###########################################################################################
'''
> Dev log (format < Date > | <Author(s)> )  
> - Developed: 30 May 2023 | JiHoon Kim <br>
> - Tested and improved: 17 July 2023 | Roshan Rane <br>
> - Tested: 28 July 2023 | JiHoon Kim <br>
> - Updated: 18 October 2023 | JiHoon Kim | Roshan Rane <br>
'''

def fit_DL_model(dataset_path,
                label,
                model = SimpleCNN(num_classes=2),
                GPUs = [1], max_epochs=10,
                show_batch=False, show_training_curves=True,
                random_seed=None, debug=False):
    
    # set GPU settings
    torch.set_float32_matmul_precision('medium')
    
    if debug: 
        os.environ["CUDA_LAUNCH_BLOCKING"]=1
        random_seed = 42
        
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
    print(f"Dataset: {dataset_path} ({unique_name})\n  Training data split = {len(df_train)} \n \
  Validation data split = {len(df_val)} \n  Test data split = {len(df_test)}")
    
    # generate data loaders
    train_loader, val_loader, test_loader = get_dataset_loaders(
                    data_split_dfs=[df_train, df_val, df_test],
                    shuffles=[True,False,False],
                    data_dir=dataset_path,
                    batch_size=64, 
                    num_workers=20, transform=[])
    if show_batch or debug:
        viz_batch(val_loader, title="Validation data")

    
    # load model
    lightning_model = LightningModel(model, learning_rate=0.05)
    logger = CSVLogger(save_dir="logs/", name=unique_name)
    
    # train model
    trainer = L.Trainer(max_epochs=max_epochs,
                        accelerator="gpu", devices=GPUs,
                        logger=logger) #deterministic=True
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
            agg[agg_col] = i
            aggreg_metrics.append(agg)
        
        f, axes = plt.subplots(1,2, sharex=True, 
                               constrained_layout=True, 
                               figsize=(7,3))
        df_metrics = pd.DataFrame(aggreg_metrics)
        df_metrics[["train_loss", "val_loss"]].plot(
            ylabel="Loss", ax=axes[0],
            grid=True, legend=True, xlabel="Epoch", 
        )
        df_metrics[["train_acc", "val_acc"]].plot(
            ylabel="Accuracy", ax=axes[1],
            grid=True, legend=True, xlabel="Epoch", ylim=(0,1)
        )
        plt.show()
        
    # test model
    train_acc = trainer.test(lightning_model, verbose=False, 
                             dataloaders=train_loader)[0]["accuracy"] #TODO use balanced accuracy?
    val_acc = trainer.test(lightning_model, verbose=False, 
                           dataloaders=val_loader)[0]["accuracy"]
    test_acc = trainer.test(lightning_model, verbose=False, 
                            dataloaders=test_loader)[0]["accuracy"]

    print(
        f"Final accuracy on Dataset: {dataset_path} ({unique_name})\n"+
        f"Train Acc {train_acc*100:.2f}"+
        f" | Val Acc {val_acc*100:.2f}"+
        f" | Test Acc {test_acc*100:.2f}"
    )
    
    return trainer, logger