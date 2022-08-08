"""
Original Source: Fenqiang Zhao, https://github.com/zhaofenqiang

"""

import torch
import numpy as np
import torch.nn as nn

from sphericalunet.utils.utils import *
from sphericalunet.layers import *


class onering_conv_layer(nn.Module):

    """
    Edits:
        Added Batching (maintains original dimension order to increase intercompatibilty)
    """

    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices=None, neigh_weights=None):
        super(onering_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders

        self.weight = nn.Linear(7 * in_feats, out_feats)

    def forward(self, x):

        mat = x[self.neigh_orders]
        mat = np.swapaxes(mat, 0, 2)
        mat = np.swapaxes(mat, 1, 2)
        mat = mat.reshape(mat.shape[0], -1,  7*self.in_feats)

        out_features = self.weight(mat)
        out_features = np.swapaxes(out_features, 1, 2)
        out_features = np.swapaxes(out_features, 0, 2)

        return out_features



class pool_layer(nn.Module):
    """

    Edits:
        Added Batching (maintains original dimension order to increase intercompatibilty)

    """

    def __init__(self, neigh_orders, pooling_type='mean'):
        super(pool_layer, self).__init__()

        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type

    def forward(self, x):

        num_nodes = int((x.size()[0]+6)/4)
        feat_num = x.size()[1]
        x = x[self.neigh_orders[0:num_nodes*7]]
        x = x.reshape(num_nodes, feat_num, 7, -1)
        x = np.swapaxes(x, 3, 2)
        x = np.swapaxes(x, 2, 1)
        x = np.swapaxes(x, 1, 0)

        if self.pooling_type == "mean":
            x = torch.mean(x, 3)
        if self.pooling_type == "max":
            x = torch.max(x, 3)
            assert(x[0].size() == torch.Size([num_nodes, feat_num]))
            return x[0], x[1]

        #assert(x.size() == torch.Size([num_nodes, feat_num]))

        x = np.swapaxes(x, 1, 0)
        x = np.swapaxes(x, 1, 2)
        return x



class upconv_layer(nn.Module):
    """
    
    Edits:
        Added Batching (maintains original dimension order to increase intercompatibilty)

    """

    def __init__(self, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        self.weight = nn.Linear(in_feats, 7 * out_feats)

    def forward(self, x):

        raw_nodes = x.size()[0]
        new_nodes = int(raw_nodes*4 - 6)

        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 0)
        x = self.weight(x)
        x = x.reshape(x.shape[0], -1, self.out_feats)
        x1 = x[ :, self.upconv_top_index]

        #assert(x1.size() == torch.Size([raw_nodes, self.out_feats]))
        x2 = x[:, self.upconv_down_index].reshape(x.shape[0], -1, self.out_feats, 2)
        x = torch.cat((x1,torch.mean(x2, 3)), 1)

        #assert(x.size() == torch.Size([new_nodes, self.out_feats]))
        x = np.swapaxes(x, 0, 1)
        x = np.swapaxes(x, 1, 2)

        return x


class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()


#        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x

