import torch
import numpy as np
import copy
import math

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)

            yield batch








class SupportsetSampler():

    def __init__(self, label, n_cls, n_per,  n_batch=1, seq_sample=False):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.seq_sample = seq_sample
        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
       
        for i in range(min(label), max(label) + 1):
           
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            assert len(self.m_ind) == self.n_cls
            if self.seq_sample:
                classes =  list(range(len(self.m_ind)))[:self.n_cls]
            else:
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                if self.seq_sample:
                    pos = list(range(len(l)))[:self.n_per]
                else:
                    pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch