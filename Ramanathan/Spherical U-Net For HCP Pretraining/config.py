import logging


batch_size_spatial      = 2048                # time steps per batch for spatial AE training
cuda                    = True               # cuda or cpu
reduce_LR_On_Plateau    = True               # lr control
lr                      = 1e-3               # learning rate
checkpoint_freq         = 5                  # epochs per checkpoint
num_epochs_spatial      = 10                 # max number of epochs to train the spatial AE for
num_epochs_temporal     = 30                 # max number of epochs to train the temporal model for
checkpoint_loc          = './checkpoints'    # location to store weigths
batch_size_temporal     = 16                 # samples per reduction
temporal_model          = 'LSTM_AE'          # 'Transformer' or 'LSTM_AE'
data_cat                = 'EMOTION'          # EMOTION, SOCIAL or REST
data_root               = '/space_lin1/hcp'  # location of subject data
logging_file            = 'log.txt'          # name for log file