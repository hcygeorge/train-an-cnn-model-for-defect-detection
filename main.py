#%%
# Working directory
import os
import argparse
import logging
import time
import pickle
import itertools
import time
from collections import Counter, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.data as Data
from load_data import CreateList, CustomDataset
from models import LeNet5, VGG, BCNN
from utils import updateBN, savemodel, Log

#%% Log setting
trial_info = 'lenet_init'  # info of trial

log = Log(trial_info)

#%% All parameters setting
para = {
    # Dataset
    'dataset': 'aoi',
    'batch_size': 48,
    'split': 0.8,  # ratio of training data
    # Model
    'resume': '',  # a path of trained model
    'pruned': '',  # a path of pruned model
    'pretrain': False,
    'cfg': [],  # None or a list of integers and 'M'
    # Training
    'cuda': True,
    'workers': 0,
    'epochs': 100,
    'checkpoint_freq': 5,
    'early_stop': False,
    # Hyperparameters
    'lr': 1e-2,
    'decay': 1e-5,
    'channel_sparsity': True,  # whether adding L1-norm of BN gamma factor
    'sparsity_rate': 0,
    'patience': 8,
    # Trial id
    'trial': trial_info}

log.log('Parameters Setting:\n{}'.format(para).replace(', ', ',\n '))

#%% Prepare data pipeline
dir_img_train = 'C:/Dataset/AOI/train_images/'
path_label_train = 'C:/Dataset/AOI/train.csv'

# Split image list and label list into train and valid.
train_list = CreateList(dir_img_train, path_label_train, shuffle=True)
train_valid_split = round(train_list.length * para['split'])
train_img = train_list.img[:train_valid_split]
train_label = train_list.label[:train_valid_split]
valid_img = train_list.img[train_valid_split:]
valid_label = train_list.label[train_valid_split:]

# Image preprocessing
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

log.log('Data Preprocessing:\n{}'.format(transform))
# Create DataLoader
train_dataset = CustomDataset(train_img,
                                train_label,
                                transform['train'])
valid_dataset = CustomDataset(valid_img,
                                valid_label,
                                transform['valid'])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=para['batch_size'],
                                        shuffle=False,
                                        num_workers=para['workers'],
                                        pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                        batch_size=para['batch_size'],
                                        shuffle=False,
                                        num_workers=para['workers'],
                                        pin_memory=True)

#%% Build a model
# net = VGG(dataset=para['dataset'], pretrained=para['pretrain'])
net = LeNet5('aoi')

# Send model into gpu memory
if para['cuda']:
    net.cuda()

log.log('Model Structure:\n{}'.format(net))

#%% Create loss function, optimzier and training scheduler
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),
                    lr=para['lr'],
                    weight_decay=para['decay'],
                    momentum=0.9,
                    nesterov=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.1, patience=para['patience'], verbose=True, threshold=1e-4, min_lr=1e-6)

log.log('Optimizer:\n{}'.format(optimizer))
#%% Train the Model
start_epoch = 0
best_prec1 = 0
if __name__ == '__main__':
    start_training = time.time()
    log.log('Start training model...')

    for epoch in range(start_epoch, start_epoch + para['epochs']):
        # loss list
        list_loss_train = []
        list_loss_valid = []
        # training
        train_correct = 0
        train_total = 0
        net.train()  # activate autograd
        for i, (images, label) in enumerate(train_loader):
            if para['cuda']:
                images, label = images.cuda(), label.cuda()
            
            optimizer.zero_grad()  # clear buffer
            out = net(images) 
            train_loss = criterion(out, label)
            train_loss.backward()  
            # subgradient decent
            if para['channel_sparsity']:
                updateBN(net, para['sparsity_rate'], False, first, last)
            optimizer.step()  # update weights
            
            _, pred = torch.max(out.data, 1)  # max() return maximum and its index in each row
            train_total += float(label.size(0))
            train_correct += float((pred == label).sum())
            
        # validation
        valid_correct = 0
        valid_total = 0
        net.eval()
        with torch.no_grad():
            for images, label in valid_loader:
                if para['cuda']:
                    images, label = images.cuda(), label.cuda()
                
                out = net(images)  # forward
                valid_loss = criterion(out, label)
                _, pred = torch.max(out.data, 1)  # max() return maximum and its index in each row
                valid_total += float(label.size(0))
                valid_correct += float((pred == label).sum())

        # metrics
        train_acc = 100*train_correct / train_total
        valid_acc = 100*valid_correct / valid_total
        is_best = valid_acc > best_prec1
        best_prec1 = max(valid_acc, best_prec1)
        list_loss_train.append(train_loss)
        list_loss_valid.append(valid_loss)

        scheduler.step(valid_acc)
        
        # save model
        state = {
            'epoch': epoch,  # last epoch
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
        state.update(para)
        suffix = para['trial']
        # save pruned structure
        if para['pruned']:
            state['cfg'] = pruned_pkl['cfg']
            suffix += '_' + args.pruned.split('_')[-1][:-4]

        save = savemodel(state, is_best, para['checkpoint_freq'], suffix, False)
        if save:
            log.log(save)

        # print result
        if (epoch+1) % 1 == 0:

            log.log('Epoch:{}/{}\nAccuracy(Train/Valid):{:.02f}/{:.02f}% Loss(Train/Valid):{:.3f}/{:.3f}'.format(
                epoch, start_epoch + para['epochs']-1, train_acc, valid_acc, train_loss, valid_loss))

        # early stopping
        if para['early_stop'] and valid_acc > 99.5:
            log.log('Early stop beacause valid accuracy > 99.5.')
            break
        
    end_training = time.time()
    log.log('Time:', round((end_training - start_training)/60, 2), 'mins')
