import os
import pandas as pd
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

def split_dataset(data , split=0.2, seed = 42):
    train_idx, test_idx = train_test_split(list(range(len(data))), test_size = split, random_state = seed)
    dataset = {}
    dataset['train'] = Subset(data, train_idx)
    dataset['test'] = Subset(data, test_idx)
    return dataset

def normalize(mat):
    mat_mean = np.mean(mat, axis=0)
    mat_std = np.std(mat, axis=0)
    mat_features = (mat - mat_mean)/mat_std
    
    return mat_features

@static_vars(file_name = 'default')
def read_file(path):
    if read_file.file_name != path:
        read_file.file_name = path
        read_file.file = normalize(mne.read_source_estimate(path).data.T)
    
    return torch.from_numpy(np.asarray(read_file.file.data))


class EmotionDataset_Spatial(Dataset):

    def __init__(self, root_dir):
        self.root = root_dir
        self.min_lin = 164
        self.folderList = os.listdir(self.root)
        self.files = []
        for folder in self.folderList:
            if os.path.exists(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_EMOTION_LR-lh.stc'):
                self.files.append(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_EMOTION_LR-lh.stc')


    def __len__(self):
        return len(self.files)*self.min_lin

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_path = self.files[idx // self.min_lin]
        self.file_data = read_file(file_path)[idx % self.min_lin]
        
        time_step = torch.zeros( 2, 10242)
        time_step[0,:] = self.file_data[:10242]
        time_step[1,:] = self.file_data[10242:]

        return time_step


class SocialDataset_Spatial(Dataset):
    
    def __init__(self, root_dir):
        self.root = root_dir
        self.min_lin = 273
        self.folderList = os.listdir(self.root)
        self.files = []
        for folder in self.folderList:
            if os.path.exists(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_SOCIAL_LR-lh.stc'):
                self.files.append(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_SOCIAL_LR-lh.stc')


    def __len__(self):
        return len(self.files)*self.min_lin

    def __getitem__(self, idx):
        file_path = self.files[idx // self.min_lin]
        self.file_data = read_file(file_path)[idx % self.min_lin]
        
        time_step = torch.zeros( 2, 10242)
        time_step[0,:] = self.file_data[:10242]
        time_step[1,:] = self.file_data[10242:]

        return time_step

class RestDataset_Spatial(Dataset):
    
    def __init__(self, root_dir):
        self.root = root_dir
        self.min_lin = 1199
        self.folderList = os.listdir(self.root)
        self.files = []
        for folder in self.folderList:
            if os.path.exists(self.root + '/' + folder + '/analysis/' + folder + '2_fsaverage_rfMRI_REST1_LR_hp2000_clean-lh.stc'):
                self.files.append(self.root + '/' + folder + '/analysis/' + folder + '2_fsaverage_rfMRI_REST1_LR_hp2000_clean-lh.stc')


    def __len__(self):
        return len(self.files)*self.min_lin

    def __getitem__(self, idx):
        file_path = self.files[idx // self.min_lin]
        self.file_data = read_file(file_path)[idx % self.min_lin]
        
        time_step = torch.zeros(2, 10242)
        time_step[0,:] = self.file_data[:10242]
        time_step[1,:] = self.file_data[10242:]

        return time_step


class EmotionDataset_Temporal(Dataset):

    def __init__(self, root_dir):
        self.root = root_dir
        self.min_lin = 164
        self.folderList = os.listdir(self.root)
        self.files = []
        for folder in self.folderList:
            if os.path.exists(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_EMOTION_LR-lh.stc'):
                self.files.append(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_EMOTION_LR-lh.stc')


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        self.file_data = read_file(file_path)
        
        sample = torch.zeros(self.min_lin, 2, 10242)
        sample[:,0,:] = self.file_data[:, :10242]
        sample[:,1,:] = self.file_data[:, 10242:]

        return sample


class SocialDataset_Temporal(Dataset):
    
    def __init__(self, root_dir):
        self.root = root_dir
        self.min_lin = 273
        self.folderList = os.listdir(self.root)
        self.files = []
        for folder in self.folderList:
            if os.path.exists(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_SOCIAL_LR-lh.stc'):
                self.files.append(self.root + '/' + folder + '/analysis/' + folder + '_2_fsaverage_tfMRI_SOCIAL_LR-lh.stc')


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        self.file_data = read_file(file_path)
        
        sample = torch.zeros(self.min_lin, 2, 10242)
        sample[:,0,:] = self.file_data[:, :10242]
        sample[:,1,:] = self.file_data[:, 10242:]

        return sample

class RestDataset_Temporal(Dataset):
    
    def __init__(self, root_dir):
        self.root = root_dir
        self.min_lin = 1199
        self.folderList = os.listdir(self.root)
        self.files = []
        for folder in self.folderList:
            if os.path.exists(self.root + '/' + folder + '/analysis/' + folder + '2_fsaverage_rfMRI_REST1_LR_hp2000_clean-lh.stc'):
                self.files.append(self.root + '/' + folder + '/analysis/' + folder + '2_fsaverage_rfMRI_REST1_LR_hp2000_clean-lh.stc')


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        self.file_data = read_file(file_path)
        
        sample = torch.zeros(self.min_lin, 2, 10242)
        sample[:,0,:] = self.file_data[:, :10242]
        sample[:,1,:] = self.file_data[:, 10242:]

        return sample