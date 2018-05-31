import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import glob
import os

def train_valid_test_loader(path, train_valid_ratio=(0.8,0.1), batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transforms = transforms.Compose([
                                 transforms.Resize((256,256)),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    untrain_transforms = transforms.Compose([
                                 transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])
    
    train_dataset = datasets.ImageFolder(path, train_transforms)
    untrain_dataset = datasets.ImageFolder(path, untrain_transforms)
    num_train = len(train_dataset)
    print("number of total examples is %d" % num_train)
    indices = list(range(num_train))
    split1 = int(np.floor(train_valid_ratio[0] * num_train))
    split2 = int(np.floor(sum(train_valid_ratio) * num_train))
    (train_idx, valid_idx, test_idx) = (indices[:split1], indices[split1:split2], indices[split2:])
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = data.DataLoader(
        untrain_dataset, batch_size=batch_size,
        sampler=valid_sampler,num_workers=num_workers, pin_memory=pin_memory)
    test_loader = data.DataLoader(
        untrain_dataset, batch_size=batch_size,
        sampler=test_sampler,num_workers=num_workers, pin_memory=pin_memory)
    print("number of batches for train, valid and test is %d, %d, %d"%(len(train_loader), len(valid_loader), len(test_loader)))

    return train_loader, valid_loader, test_loader 

def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)