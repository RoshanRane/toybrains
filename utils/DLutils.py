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
    

def get_dataset_loaders(data_split_dfs,
                        data_dir="toybrains", 
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
    # (TODO) Validation and Test should be set shuffle=False (wrong imlementation)
    data_loaders = []
    
    for data_split_df in data_split_dfs:
        dataset = ToyBrainsDataset(
                    df=data_split_df,
                    img_dir=f'{data_dir}/images',
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
