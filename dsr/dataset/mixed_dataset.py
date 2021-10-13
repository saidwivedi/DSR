"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from loguru import logger
from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, method, **kwargs):

        if options.USE_ALL_ITW:
            self.dataset_list = ['h36m', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
            self.dataset_dict = {'h36m': 0, 'mpii': 1, 'lspet': 2, 'coco': 3, 'mpi-inf-3dhp': 4}
        else:
            if options.TRAIN_3DPW:
                self.dataset_list = ['h36m', 'coco', 'mpi-inf-3dhp', '3dpw']
                self.dataset_dict = {'h36m': 0, 'coco': 1, 'mpi-inf-3dhp': 2, '3dpw': 3}
            else:
                self.dataset_list = ['h36m', 'coco', 'mpi-inf-3dhp']
                self.dataset_dict = {'h36m': 0, 'coco': 1, 'mpi-inf-3dhp': 2}

        logger.info(f'Datasets used for training --> {self.dataset_list}')
        self.datasets = [BaseDataset(options, method, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])

        """
        Data distribution inside each batch - EFT data settings:
        50% H36M - 30% ITW - 20% MPI-INF
        """
        if options.USE_ALL_ITW:
            self.partition = [.5, .3*len(self.datasets[1])/length_itw,
                              .3*len(self.datasets[2])/length_itw,
                              .3*len(self.datasets[3])/length_itw, 
                              .2]
        else:
            if options.TRAIN_3DPW:
                self.partition = [.3, .4, .1, .2]
            else:
                self.partition = [.5, .3, .2]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.dataset_list)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
