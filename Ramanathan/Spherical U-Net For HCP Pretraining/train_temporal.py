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
    _Dataset = EmotionDataset_Temporal(config.data_root)
elif config.data_cat == 'SOCIAL':
    _Dataset = SocialDataset_Temporal(config.data_root)
elif config.data_cat == 'REST':
    _Dataset = RestDataset_Temporal(config.data_root)
else:
    print(f" *{config.data_cat}* category of data not handled..")
    config.logging.debug(f" *{config.data_cat}* category of data not handled..")
    exit()

_Dataset = split_dataset(_Dataset, 0.2)

train_dataloader = torch.utils.data.DataLoader(_Dataset['train'], batch_size=config.batch_size_spatial, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(_Dataset['test'], batch_size=config.batch_size_spatial, shuffle=False, pin_memory=True)

def batch_train(model, data):
    model.train()

    prediction, _, _ = model(data.reshape(data.shape[0], -1).float())
    
    loss = criterion(prediction, data.reshape(data.shape[0], -1).float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def batch_val(model, data):

    model.eval()

    prediction, _, _= model(data.reshape(data.shape[0], -1).float())
    
    loss = criterion(prediction, data.reshape(data.shape[0], -1).float())

    return loss.item()

def train_(model):
    val_loss_low = 1e9
    for epoch_idx in range(config.num_epochs_spatial):

        train_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):
            
            data = data.to(device)
            _ , data = spatial_model(data.T)
            data = data.T
            batch_loss = batch_train(model, data)
            train_loss += batch_loss
            print(f'Epoch {epoch_idx} [{batch_idx+1}/{len(train_dataloader)}]:: MSE: {batch_loss}')
            
            config.logging.debug(f'Epoch {epoch_idx} [{batch_idx+1}/{len(train_dataloader)}]:: MSE: {batch_loss}')
        
        val_loss = 0.0
        for batch_idx, data in enumerate(val_dataloader):
            
            data = data.to(device)
            _ , data = spatial_model(data.T)
            data = data.T
            batch_loss = batch_val(model, data)
            val_loss += batch_loss
        
        if config.reduce_LR_On_Plateau:
            scheduler.step(val_loss/len(val_dataloader))
        
        print(f'Epoch {epoch_idx}:: MSE_train: {train_loss/len(train_dataloader)} MSE_val: {val_loss/len(val_dataloader)}')
        config.logging.debug(f'Epoch {epoch_idx}:: MSE_train: {train_loss/len(train_dataloader)} MSE_val: {val_loss/len(val_dataloader)}')

        if epoch_idx % config.checkpoint_freq == 0:
            torch.save(model.state_dict(), config.checkpoint_loc + f'/{epoch_idx}_temporal_weights.pth')
        
        if val_loss < val_loss_low:
            torch.save(model.state_dict(), config.checkpoint_loc + f'/best_temporal_weights.pth')
            val_loss_low = val_loss


spatial_model = Unet(2,2,6)
spatial_model.load_state_dict(torch.load(config.checkpoint_loc + '/best_weights.pth'))
spatial_model.to(device=device)
spatial_model.eval()

if config.temporal_model == 'LSTM_AE':
    model = VAENet(in_size=64*42, out_time = _Dataset.min_len).to(device=device)
# elif config.temporal_model == 'Transformer':
#     model = TransformerNet().to(device=device) ## model needs to be configured
else:
    print(f" *{config.temporal_model}* model not handled..")
    config.logging.debug(f" *{config.temporal_model}* model not handled..")
    exit()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=3, threshold=1e-3, threshold_mode='rel', min_lr=1e-5)

train_(model)