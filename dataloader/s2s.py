import os
import os.path as osp
import re
import json
import time
import h5py
from matplotlib.font_manager import json_dump
import numpy as np
import random
import librosa
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
import pandas as pd
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class S2S(Dataset):

    def __init__(self, dataset='f2n',root='./', phase='train', 
                 index_path=None, index=None, k=5, base_sess=None, data_type='audio', args=None):
        self.root = os.path.expanduser(root)
        self.root = root
        self.data_type = data_type
        # self.make_extractor()
        self.phase = phase
        if dataset[0] == 'f':
            self.frames = 44100
        elif dataset[0] == 'l':
            self.frames = 32000
        elif dataset[0] == 'n':
            self.frames = 64000
        # self.train = train  # training set or test set
        self.all_train_df = pd.read_csv(f"/new_data/syj/CD-FCAC/data/s2s/{dataset}_fscil_train.csv")
        self.all_val_df = pd.read_csv(f"/new_data/syj/CD-FCAC/data/s2s/{dataset}_fscil_val.csv")
        self.all_test_df = pd.read_csv(f"/new_data/syj/CD-FCAC/data/s2s/{dataset}_fscil_test.csv")
        if phase == 'train':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
            else:
                """
                if shuffle_dataset:
                    random.seed(time.time)
                else:
                    random.seed(0)
                """
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
        elif phase == 'val':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=k)
        elif phase =='test':
            self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        


    def SelectfromClasses(self, df, index, per_num=None):
        """
        select k samples from list_dict which label=index 
        Args:
            paths (list):
            labels (list):
            index (list):
            k (int): 
        
        Returns:
            data_tmp (list):
            target_tmp (list):
        """
        data_tmp = []
        targets_tmp = []

        for i in index:
            ind_cl = np.where(i == df['label'])[0]
            # random.shuffle(ind_cl)

            k = 0
            # ind_cl is the index list whose label equals i, 
            # start_idx make sure 
            # there is no intersection between train and test set  
            for j in ind_cl:
                data_tmp.append(df['path'][j])
                targets_tmp.append(df['label'][j])
                k += 1
                if per_num is not None:
                    if k >= per_num:
                        break
              
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        # feature = self.wave_to_tfr(path)
        # feature = self.wave_to_logmel(path)
        audio,sr = torchaudio.load(path)
        # if len(audio) < 64000:
            # audio = torch.concat([audio, torch.zeros([1, 64000 - audio.shape[1]], dtype=audio.dtype)], dim=1)
        audio = torch.concat([audio, audio], dim=1)
        audio = audio[:, :self.frames]
        return audio, targets

if __name__ == '__main__':

    # class_index = open(txt_path).read().splitlines()
    base_class = 60
    class_index = np.arange(base_class, 100)
    dataroots = ["/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data",
                "/data/datasets/The_NSynth_Dataset",
                "/data/datasets/librispeech_fscil/spk_segments"]
    batch_size_base = 400
    trainset = S2S(across_dataset='f2n', root=dataroots, phase="train",  index=class_index, k=5,
                      base_sess=False)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    list(trainloader)    
