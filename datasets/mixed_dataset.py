"""
# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/mixed_dataset.py
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np
from pare.dataset.coco_occlusion import load_pascal_occluders

from core.path_config import PASCAL_ROOT
from .base_dataset import BaseDataset


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, options, **kwargs):
        # self.dataset_list = ['h36m', 'agora', 'spec-syn', 'humman', 'mpi-inf-3dhp', 'coco_cliff', 'lspet',
        #                      'lsp-orig', '3dpw', 'mpii_cliff']
        # # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
        # self.dataset_dict = {'h36m': 0, 'agora': 1, 'spec-syn': 2, 'humman': 3, 'mpi-inf-3dhp': 4,
        #                      'coco_cliff': 5, 'lspet': 6, 'lsp-orig': 7, '3dpw': 8, 'mpii_cliff': 9}
        #
        # self.occluders = load_pascal_occluders(pascal_voc_root_path=PASCAL_ROOT)
        # self.datasets = [BaseDataset(options, ds, self.occluders, **kwargs) for ds in self.dataset_list]
        #
        # syn_length = sum([len(ds) for ds in self.datasets[1:3]])
        # coco_length = sum([len(ds) for ds in self.datasets[5:6]])
        # mix_length = sum([len(ds) for ds in self.datasets[7:]])
        # self.length = 250000
        # """
        # Data distribution inside each batch:
        # 30% H36M - 60% ITW - 10% MPI-INF
        # """
        # self.partition = [
        #     .4,
        #     .2 * len(self.datasets[1]) / syn_length,
        #     .2 * len(self.datasets[2]) / syn_length,
        #     .1,
        #     .1,
        #     .1 * len(self.datasets[5]) / coco_length,
        #     .05,
        #     .05 * len(self.datasets[7]) / mix_length,
        #     .05 * len(self.datasets[8]) / mix_length,
        #     .05 * len(self.datasets[9]) / mix_length, ]
        # self.partition = np.array(self.partition).cumsum()

        # self.dataset_list = ['h36m', 'mpii-vitpose', 'coco-pruned', 'coco-vitpose-pruned', 'mpi-inf-pruned', 'ava',
        #                      'aic', 'insta']
        # self.dataset_dict = {'h36m': 0, 'mpii-vitpose': 1, 'coco-pruned': 2, 'coco-vitpose-pruned': 3,
        #                      'mpi-inf-pruned': 4, 'ava': 5, 'aic': 6, 'insta': 7}
        #
        # self.occluders = load_pascal_occluders(pascal_voc_root_path=PASCAL_ROOT)
        # self.datasets = [BaseDataset(options, ds, self.occluders, **kwargs) for ds in self.dataset_list]
        #
        # self.length = 180000
        # self.partition = [
        #     .1,
        #     .1,
        #     .1,
        #     .1,
        #     .02,
        #     .19,
        #     .19,
        #     .2,
        # ]
        # self.partition = np.array(self.partition).cumsum()

        self.dataset_list = ['h36m', 'mpii-vitpose', 'coco-pruned', 'coco-vitpose-pruned', 'mpi-inf-pruned', 'ava',
                             'aic', 'insta', 'agora_1280x720', '3dpw', 'humman']
        self.dataset_dict = {'h36m': 0, 'mpii-vitpose': 1, 'coco-pruned': 2, 'coco-vitpose-pruned': 3,
                             'mpi-inf-pruned': 4, 'ava': 5, 'aic': 6, 'insta': 7, 'agora_1280x720': 8, '3dpw': 9,
                             'humman': 10}

        self.occluders = load_pascal_occluders(pascal_voc_root_path=PASCAL_ROOT)
        self.datasets = [BaseDataset(options, ds, self.occluders, **kwargs) for ds in self.dataset_list]

        self.length = 165000
        self.partition = [
            .07,
            .05,
            .05,
            .05,
            .02,
            .18,
            .18,
            .19,
            .07,
            .07,
            .07,
        ]
        self.partition = np.array(self.partition).cumsum()

        # self.dataset_list = ['h36m', 'mpii-vitpose', 'coco-pruned', 'coco-vitpose-pruned', 'mpi-inf-pruned', 'ava',
        #                      'aic', 'insta', 'agora_1280x720', 'spec-syn', 'humman']
        # self.dataset_dict = {'h36m': 0, 'mpii-vitpose': 1, 'coco-pruned': 2, 'coco-vitpose-pruned': 3,
        #                      'mpi-inf-pruned': 4, 'ava': 5, 'aic': 6, 'insta': 7, 'agora_1280x720': 8, 'spec-syn': 9,
        #                      'humman': 10}
        #
        # self.occluders = load_pascal_occluders(pascal_voc_root_path=PASCAL_ROOT)
        # self.datasets = [BaseDataset(options, ds, self.occluders, **kwargs) for ds in self.dataset_list]
        #
        # self.length = max([len(ds) for ds in self.datasets])
        # self.partition = [
        #     .07,
        #     .06,
        #     .06,
        #     .06,
        #     .02,
        #     .19,
        #     .19,
        #     .2,
        #     .07,
        #     .01,
        #     .07,
        # ]
        # self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(11):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
