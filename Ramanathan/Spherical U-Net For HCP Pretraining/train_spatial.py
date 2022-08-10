import torch
import torch.nn as nn
import config
import os
from layer_mods import *
from models import *
from Data_Loader import *


config.logging.basicConfig(filename=config.logging_file, level=config.logging.DEBUG)

if not os.path.exists(config.checkpoint_loc):
    os.mkdir(config.checkpoint_loc)

if config.cuda and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if config.data_cat == 'EMOTION':
    _Dataset = EmotionDataset_Spatial(config.data_root)
elif config.data_cat == 'SOCIAL':
    _Dataset = SocialDataset_Spatial(config.data_root)
elif config.data_cat == 'REST':
    _Dataset = RestDataset_Spatial(config.data_root)
else:
    print(f" *{config.data_cat}* category of data not handled..")
    config.logging.debug(f" *{config.data_cat}* category of data not handled..")
    exit()

_Dataset = split_dataset(_Dataset, 0.2)

train_dataloader = torch.utils.data.DataLoader(_Dataset['train'], batch_size=config.batch_size_spatial, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(_Dataset['test'], batch_size=config.batch_size_spatial, shuffle=False, pin_memory=True)

def batch_train(model, data):
    model.train()
    data = data.to(device)

    prediction, _ = model(data.reshape(data.shape[0], 2, -1).float().T)
    
    loss = criterion(prediction, data.reshape(data.shape[0], 2, -1).float().T)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def batch_val(model, data):

    model.eval()
    data = data.to(device)

    prediction, _ = model(data.reshape(data.shape[0], 2, -1).float().T)
    
    loss = criterion(prediction, data.reshape(data.shape[0], 2, -1).float().T)

    return loss.item()

def train_(model):
    val_loss_low = 1e9
    for epoch_idx in range(config.num_epochs_spatial):

        train_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):

            batch_loss = batch_train(model, data)
            train_loss += batch_loss
            print(f'Epoch {epoch_idx} [{batch_idx+1}/{len(train_dataloader)}]:: MSE: {batch_loss}')
            
            config.logging.debug(f'Epoch {epoch_idx} [{batch_idx+1}/{len(train_dataloader)}]:: MSE: {batch_loss}')
        
        val_loss = 0.0
        for batch_idx, data in enumerate(val_dataloader):
            batch_loss = batch_val(model, data)
            val_loss += batch_loss
        
        if config.reduce_LR_On_Plateau:
            scheduler.step(val_loss/len(val_dataloader))
        
        print(f'Epoch {epoch_idx}:: MSE_train: {train_loss/len(train_dataloader)} MSE_val: {val_loss/len(val_dataloader)}')
        config.logging.debug(f'Epoch {epoch_idx}:: MSE_train: {train_loss/len(train_dataloader)} MSE_val: {val_loss/len(val_dataloader)}')

        if epoch_idx % config.checkpoint_freq == 0:
            torch.save(model.state_dict(), config.checkpoint_loc + f'/{epoch_idx}_weights.pth')
        
        if val_loss < val_loss_low:
            torch.save(model.state_dict(), config.checkpoint_loc + f'/best_weights.pth')
            val_loss_low = val_loss


model = Unet(2, 2, 6).to(device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=3, threshold=1e-3, threshold_mode='rel', min_lr=1e-5)

train_(model)