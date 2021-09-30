import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from common.functions import sig2spec

class TrainValDataset(Dataset):

    def __init__(self, path_name, type_, norm=True):

        self.path = Path(path_name)
        self.csv_path_list = []
        self.norm = norm

        for dir_name in self.path.glob('*/*.csv'):
            self.csv_path_list.append(str(dir_name))

        # 種目に応じてラベルを設定
        if type_ == 'thighs':
            self.label = (
                'thighs_cant_keep',
                'thighs_cant_raise',
                'thighs_cant_slow',
                'thighs_cant_stop',
                'thighs_cant_strech',
                'thighs_correct',
            )
        elif type_ == 'abs':
            self.label = (
                'abs_correct',
                'abs_keep_shoulders',
                'abs_raise',
                'abs_recoil',
            )
        elif type_ == 'shoulders':
            self.label = (
                'shoulders_correct',
                'shoulders_pull',
                'shoulders_sides',
                'shoulders_trunk',
            )

    def __len__(self):
        return len(self.csv_path_list)

    def __getitem__(self, idx):

        sig = np.loadtxt(self.csv_path_list[idx], delimiter=',', dtype='float32')

        sig_ = (sig - 512) / 1023

        if self.norm is True:
            sig_ = sig / 1023
        else:
            sig_ = sig

        spec_list = []

        for i in range(sig.shape[1]):
            spec = sig2spec(sig_[:, i], 128, 64)
            spec_list.append(spec)

        spec_ = np.array(spec_list)
        spec_ = np.abs(spec_).astype('float32')

        d, _ = os.path.split(self.csv_path_list[idx])
        _, classes = os.path.split(d)

        label = self.label.index(classes)

        return spec_, label