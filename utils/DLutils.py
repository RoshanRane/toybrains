import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# deep learning imports
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchmetrics import Metric
import torchvision
from torchvision import transforms
# custom imports 
from utils.metrics import _explained_deviance

###########################################################################################
############################          Dataloader         ##################################
###########################################################################################

class ToyBrainsDataloader(Dataset):
    def __init__(self, img_dir, img_names, labels, transform=transforms.ToTensor()):
        self.img_dir = img_dir
        self.img_names = img_names
        self.labels = labels
        # if labels are encoded as categorical names (str) then convert to integer-encoding
        self.label_enc = None
        if isinstance(self.labels[0], str):
            enc = LabelEncoder().fit(self.labels)
            # save the encoding as a dict
            self.label_enc = {i:enc.classes_[i] for i in range(len(enc.classes_))}
            self.labels = enc.transform(self.labels)
        
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
        return _explained_deviance(
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
    def __init__(self, num_classes, 
                act_size_penultimate=64,
                act_size_antepenultimate=256):
        super().__init__()
        self.act_size_penultimate = act_size_penultimate # weights + 1 bias
        self.act_size_antepenultimate = act_size_antepenultimate # weights + 1 bias
        # convolutional layers
        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32),
            nn.Dropout(0.1),
            ConvBlock(in_channels=32, out_channels=64),
            nn.Dropout(0.1),
            ConvBlock(in_channels=64, out_channels=128),
            nn.Dropout(0.1),
        )
        
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            # TODO Hardcoded: expected output of self.conv() would be 8 x 8 if input is 64 x 64
            nn.Linear(128 * 8 * 8, self.act_size_antepenultimate, bias=True),
            nn.Dropout(0.1),
            nn.Linear(self.act_size_antepenultimate, self.act_size_penultimate, bias=True),
            nn.Dropout(0.1),
            nn.Linear(self.act_size_penultimate, num_classes, bias=True),
        )

    def forward(self, x):
        h = self.conv(x)
        y = self.fc(h)
        return y

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
    
    
