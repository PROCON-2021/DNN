import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class TrainValDataset(Dataset):

    def __init__(self, path_name, type_):

        self.path = Path(path_name)
        self.csv_path_list = []

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
        sig_norm = sig / 1023

        d, _ = os.path.split(self.csv_path_list[idx])
        _, classes = os.path.split(d)

        label = self.label.index(classes)

        return sig_norm[np.newaxis,:,:], label