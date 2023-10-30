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

# import monai

# custom imports 
from utils.dataset import split_dataset
from utils.metrics import explained_deviance

###########################################################################################
############################          Dataloader         ##################################
###########################################################################################

class ToyBrainsDataloader(Dataset):
    def __init__(self, img_dir, img_names, labels, transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        number = str(self.img_names[index]).zfill(5)
        img = Image.open(os.path.join(self.img_dir, number + ".jpg"))
        
        if self.transform is not None:
            img = self.transform(img)
            
        label = torch.as_tensor(self.labels[index]).type(torch.LongTensor)
        return img, label
    
    def __len__(self):
        return self.labels.shape[0]
    
    
def get_toybrain_dataloader(
                        data_df,
                        images_dir="toybrains/images", 
                        batch_size=16, 
                        shuffle=True, 
                        num_workers=30, transform=[],
                        ):
    ''' Creates pytorch dataloader of the ToyBrainsDataloader'''
    dataset = ToyBrainsDataloader(
                img_names = data_df["subjectID"].values, # TODO change hardcoded
                labels = data_df["label"].values,
                img_dir=images_dir,
                transform=transforms.Compose(
                    [transforms.ToTensor()]+transform))
    
    data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers)
    return data_loader
    

################################################################################
####################       LIGHTNING  helpers    ###############################
################################################################################


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
            nn.Linear(3, num_classes, bias=False),
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
    
    
