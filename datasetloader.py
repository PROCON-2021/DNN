import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class MyDataset(Dataset):
    def __init__(self, path, frame_range=20, key='train'):

        # ! TODO
        # ! テスト用の処理を実装
        # ! シリアルで送られてくる信号をバッファして云々

        patience_path = 'dataset/dry_signals/patience'
        trainee_path = 'dataset/dry_signals/trainee'
        #dir = Path(path)
        patience_file_list = list(Path(patience_path).iterdir())
        trainee_file_list  = list(Path(trainee_path).iterdir())

        # 入力データの範囲
        self.frame_range = frame_range

        # 配列確保用
        tmp = np.loadtxt(patience_file_list[0], delimiter=',', dtype=np.float32)

        #クラス分類数（patience, trainee）
        class_num = 2

        # データとラベルを格納する配列
        # data : [siglen x ch x n_files]
        # label: [siglen x label x n_files]
        self.patience_data  = np.zeros([tmp.shape[0], tmp.shape[1]-2, len(patience_file_list)], dtype = "float32")
        self.patience_label = np.zeros([tmp.shape[0], class_num     , len(patience_file_list)], dtype = "float32")
        self.trainee_data   = np.zeros([tmp.shape[0], tmp.shape[1]-2, len(trainee_file_list)],  dtype = "float32")
        self.trainee_label  = np.zeros([tmp.shape[0], class_num     , len(trainee_file_list)],  dtype = "float32")

        # tqdm設定
        proc = tqdm(total=len(trainee_file_list), desc='Import dataset and label')
        for i, file in enumerate(trainee_file_list):
            #時間軸以外を取得
            self.trainee_data[:,:,i] = np.loadtxt(file, delimiter=',', dtype=np.float32)[:,1:4]
            # trainee : 0
            self.trainee_label[:, :,i] = torch.eye(class_num)[0]
            proc.update()

        # tqdm設定
        proc = tqdm(total=len(patience_file_list), desc='Import dataset and label')
        for i, file in enumerate(patience_file_list):
            #時間軸以外を取得
            self.patience_data[:,:,i] = np.loadtxt(file, delimiter=',', dtype=np.float32)[:,1:4]

            # patience: 1
            self.patience_label[:, :,i] = torch.eye(class_num)[1]
            proc.update()

        #cancatenate
        self.data = np.concatenate([self.patience_data, self.trainee_data], 2)
        self.label = np.concatenate([self.patience_label, self.trainee_label], 2)

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, idx):
        # データ全部を渡す
        # シフトしながらDNNに入力するのは別の所で実装（だと思っている）
        return self.data[:,:,idx], self.label[:,:,idx]