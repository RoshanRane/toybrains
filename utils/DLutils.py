import os
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision


#################################################################################################
# DATA MODULE
#################################################################################################

# functions

def generate_dataset(raw_csv_path, label, random_seed=42, debug=False):
    '''
    generate the dataset
    
    PARAMETERS
    ----------
    raw_csv_path : string
        csv path
        
    label : string
        labels, 'lblbin_stop-smidl-bvol', 'lblbin_stop-smidl-vthick', 'lblbin_bvol-vthick'
    
    random_seed : integer, default : 42
        random seed number
    
    NOTE
    ----
    (TODO) change function to class
    (TODO) Refactoring `torch.utils.data.random_split` or support K-fold or stratified
    
    '''
    # set random seed
    seed = random_seed
    
    # set raw csv path
    raw_csv_path = raw_csv_path
    
    # set target label
    label = label
    
    # load the raw csv
    DF = pd.read_csv(raw_csv_path)
    
    # assign target label
    DF['label'] = DF[label].astype(int)
    
    # split dataset into 80% for training and 20% for remaining
    DF_train, DF_remaining = train_test_split(DF, test_size=0.2, random_state=seed)
    
    # split remaining 20% into 10% for validation and 10% for test
    DF_val, DF_test = train_test_split(DF_remaining, test_size=0.5, random_state=seed)
    
    # reset the index
    DF_train.reset_index(inplace=True, drop=True)
    DF_val.reset_index(inplace=True, drop=True)
    DF_test.reset_index(inplace=True, drop=True)
    
    # print the number of rows in each dataframe
    if debug:
        print(f"Raw:   {len(DF)}\n"
              f"Train:  {len(DF_train)}\n"
              f"Val:    {len(DF_val)}\n"
              f"Test:   {len(DF_test)}")
    
    return DF_train, DF_val, DF_test
    

def get_dataset_loaders(data_split_dfs,
                        data_dir="toybrains/images", 
                        batch_size=16, shuffle=True, 
                        num_workers=0, transform=[],
                        ):
    ''' Creates pytorch dataloaders of the ToyBrainsDataset for all the 
    dataframes passed in *args
    
    NOTES
    ----
    reference: https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit05-lightning/5.5-datamodules
    TODO: refactoring ToyBrainsDataModule
    '''
    data_loaders = []
    
    for data_split_df in data_split_dfs:
        dataset = ToyBrainsDataset(
                    df=data_split_df,
                    img_dir=data_dir,
                    transform=transforms.Compose(
                        [transforms.ToTensor()]+transform)
                    )
        data_loader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
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
            
        label = self.labels[index]
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
    
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 2)
    
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
        return logits


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
    plt.title(title+f"\nlabels={lbls}")
    plt.imshow(
        np.transpose(
        torchvision.utils.make_grid(
            imgs, padding=1, pad_value=1.0, normalize=True), 
        (1, 2, 0)))
    plt.show()
