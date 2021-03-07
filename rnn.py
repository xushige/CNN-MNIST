import torch
import torchvision
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np

def get_trans():
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ])
    return trans

train_data = torchvision.datasets.MNIST(root="./mnist",train=True,download=False,transform=get_trans())
test_data = torchvision.datasets.MNIST(root="./mnist",train=False,download=False,transform=get_trans())


B_S = 128
train_loader = DataLoader(train_data,batch_size=B_S,shuffle=True)
test_loader = DataLoader(test_data,batch_size=B_S*10,shuffle=True)

print(len(train_data))
for data, label in train_loader:
    print(data.shape)
    print(label.shape)
    break

print(np.array(64))