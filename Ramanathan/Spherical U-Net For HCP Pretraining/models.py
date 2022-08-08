"""
Original Source: Fenqiang Zhao, https://github.com/zhaofenqiang

"""

from layer_mods import *

import torch
import numpy as np
import torch.nn as nn

class Unet(nn.Module):
       
    def __init__(self, in_ch, out_ch, level=7, n_res=5, rotated=0):

        super(Unet, self).__init__()
        
        neigh_orders = Get_neighs_order(rotated)
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*4)
        
        conv_layer = onering_conv_layer
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
      
        self.up = nn.ModuleList([])
        for i in range(n_res-1):
            self.up.append(up_block(conv_layer, chs[n_res-i], chs[n_res-1-i],
                                    neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
            
        self.outc = nn.Linear(chs[1], out_ch)
                
        self.n_res = n_res
        
    def forward(self, x):
        
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        x = xs[-1]
        
        for i in range(self.n_res-1):
            x = self.up[i](x, xs[self.n_res-1-i])
        
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 0)
        x = self.outc(x)
        x = np.swapaxes(x, 1, 0)
        x = np.swapaxes(x, 1, 2)
        return x


mse = nn.MSELoss()

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)
  
  def forward(self, x):

    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

class TransformerNet(nn.Module):
  def __init__(self, in_size, seq_len, emb_size, num_heads):

    super(TransformerNet, self).__init__()

    self.positional_encoding_layer = PositionalEncoding( d_model=emb_size, dropout=0.05)

    encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, 
            nhead= num_heads,
            dim_feedforward=emb_size,
            dropout=0.05,
            batch_first=True 
            )
    
    self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers= 3
            )
    
    decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead= num_heads,
            dim_feedforward=emb_size,
            dropout=0.05,
            batch_first=True
            )
    
    self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=3
            )
    
    self.out_layer = nn.Linear(emb_size, in_size)
    self.enc_in_layer = nn.Linear(in_size, emb_size)
    self.dec_in_layer = nn.Linear(in_size, emb_size)
    self.tanh = nn.Tanh()

      
  def forward(self, x):

    dec_inp = torch.zeros_like(x)

    emb = self.encoder(self.positional_encoding_layer(self.enc_in_layer(x)))

    out = self.decoder(tgt = self.dec_in_layer(dec_inp), memory = emb)

    out = self.tanh(self.out_layer(out))
    

    return out, emb



# https://machinelearningmastery.com/lstm-autoencoders/

def recon_Loss(out, gt):
  return nn.functional.mse_loss(out, gt)

def KLD_Loss(mean, log_var):
  return torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(),dim=1),dim=0)

def VAE_Loss(out, gt):
  return recon_Loss(out[0], gt) + KLD_Loss(out[1], out[2])


class VAEEnc(nn.Module):
  def __init__(self, in_size):
    super(VAEEnc, self).__init__()

    self.en_lstm1 = nn.LSTM(in_size, 256, 1, batch_first=True)
    self.en_lstm2 = nn.LSTM(256, 128, 1, batch_first=True)
    self.en_fc1 = nn.Linear(128, 64)
    self.en_fc2 = nn.Linear(128, 64)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    batch_size = x.size(0)
    
    h_0 = torch.zeros(1, batch_size, 256)
    c_0 = torch.zeros(1, batch_size, 256)
    
    out , (final_hidden_state, final_cell_state) = self.en_lstm1(x, (h_0, c_0))
    out = self.tanh(out)

    h_1 = torch.zeros(1, batch_size, 128)
    c_1 = torch.zeros(1, batch_size, 128)
    
    out , (final_hidden_state, final_cell_state) = self.en_lstm2(out, (h_1, c_1))
    out = self.tanh(out)
    out = out[:,-1,:]

    mean = self.sigmoid(self.en_fc1(out))
    log_var = self.sigmoid(self.en_fc2(out))

    return mean, log_var


class VAEDec(nn.Module):
  def __init__(self, in_size, out_time):
    super(VAEDec, self).__init__()

    self.in_size = in_size
    self.time = out_time
    self.de_lstm1 = nn.LSTM(128, 256, 1, batch_first=True)
    self.de_fc1 = nn.Linear(64, 128)
    self.de_TDD = nn.ModuleList([nn.Linear(256, in_size) for i in range(out_time)])

    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    batch_size = x.size(0)

    out = self.tanh(self.de_fc1(x))
    
    out = torch.zeros((batch_size, self.time, out.size(1)), requires_grad=True) + out.reshape(batch_size, 1, -1)

    h_0 = torch.zeros(1, batch_size, 256)
    c_0 = torch.zeros(1, batch_size, 256)
    
    out , (final_hidden_state, final_cell_state) = self.de_lstm1(out, (h_0, c_0))
    out = self.tanh(out)
    out_TDD = torch.zeros((out.size(0), out.size(1), self.in_size))
    for i in range(self.time):
      out_TDD[:, i, :] = self.de_TDD[i](out[:,i,:])

    return out_TDD

class VAENet(nn.Module):
  def __init__(self, in_size, out_time):

    super(VAENet, self).__init__()

    self.enc = VAEEnc(in_size)
    self.dec = VAEDec(in_size, out_time)


  def reparameterise(self, mean, log_var):

    if self.training:
      std = torch.exp(0.5*log_var)
      eps = std.data.new(std.size()).normal_()
      return eps*std + mean
    else:
      return mean
      
  def forward(self, x):

    mean, log_var = self.enc(x)
    rep = self.reparameterise(mean, log_var)
    out = self.dec(rep)

    return out, mean, log_var