import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, path_name):

        self.path = Path(path_name)
        self.csv_path_list = []

        for dir_name in self.path.glob('*/*.csv'):
            self.csv_path_list.append(str(dir_name))

    def __len__(self):
        return len(self.csv_path_list)

    def __getitem__(self, idx):
        raise NotImplementedError

class ThighsTrainValDataset(BaseDataset):

    def __init__(self, path_name):
        super().__init__(path_name)

        self.label = (
            'thighs_cant_keep',
            'thighs_cant_raise',
            'thighs_cant_slow',
            'thighs_cant_stop',
            'thighs_cant_strech',
            'thighs_correct',
        )

    def __getitem__(self, idx):

        sig = np.loadtxt(self.csv_path_list[idx], delimiter=',', dtype='float32')
        sig_norm = sig / 1023

        d, _ = os.path.split(self.csv_path_list[idx])
        _, classes = os.path.split(d)

        label = self.label.index(classes)

        return sig_norm[np.newaxis,:,:], label