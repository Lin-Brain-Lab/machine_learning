import torch
import torch.nn as nn
import numpy as np
import os
import config
from layer_mods import *
from models import *
from Data_Loader import *
from torch.utils.data import DataLoader
import logging


if config.cuda and torch.cuda.is_available():
    device = 'cuda:0'
else
    device = 'cpu

if data_cat == 'EMOTION':
    Dataset = EmotionDataset_Spatial(config.data_root)
elif data_cat == 'SOCIAL':
    Dataset = SocialDataset_Spatial(config.data_root)
elif data_cat == 'REST':
    Dataset = RestDataset_Spatial(config.data_root)
else:
    print(f" *{data_cat}* category of data not handled..")
    exit()

Dataset = split_dataset(Dataset, 0.2)

train_dataloader = torch.utils.data.DataLoader(Dataset['train'], batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(Dataset['test'], batch_size=batch_size, shuffle=False, pin_memory=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=3, verbose=True, threshold=0.001, threshold_mode='rel', min_lr=0.000001)

def batch_train(model, data):
    model.train()
    data = data.to(device)

    prediction = model(data.reshape(data.shape[0], 2, -1).float().T)
    
    loss = criterion(prediction, data.reshape(data.shape[0], 2, -1).float().T)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def batch_val(model, data):

    model.eval()
    data = data.to(device)

    prediction = model(data.reshape(data.shape[0], 2, -1).float().T)
    
    loss = criterion(prediction, data.reshape(data.shape[0], 2, -1).float().T)

    return loss.item()

def train_():

    model = Unet(2, 2, 6)

    for epoch_idx in range(config.num_epochs_spatial):

        train_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):

            batch_loss = batch_train(model, data)
            train_loss += batch_loss
            print(f'Epoch {epoch_idx} [{batch_idx+1}/{len(train_dataloader)}]:: MSE: {batch_loss}')
        
        val_loss = 0.0
        for batch_idx, data in enumerate(val_dataloader):
            batch_loss = batch_val(model, data)
            val_loss += batch_loss
        
        print(f'Epoch {epoch_idx}:: MSE_train: {train_loss} MSE_val: {val_loss}')

