import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class MyDataset(Dataset):
    def __init__(self, path, frame_range=20, key='train'):

        # ! TODO
        # ! テスト用の処理を実装
        # ! シリアルで送られてくる信号をバッファして云々

        dir = Path(path)
        file_list = list(dir.iterdir())

        # 入力データの範囲
        self.frame_range = frame_range

        # 配列確保用
        tmp = np.loadtxt(file_list[0], delimiter=',', dtype=np.float32)

        # データとラベルを格納する配列
        # data : [siglen x ch x n_files]
        # label: [siglen x label x n_files]
        self.data = np.zeros([tmp.shape[0], tmp.shape[1], len(file_list)])
        self.label = np.zeros([tmp.shape[0], 1, len(file_list)])

        # tqdm設定
        proc = tqdm(total=len(file_list), desc='Import dataset and label')

        for i, file in enumerate(file_list):
            self.data[:,:,i] = np.loadtxt(file, delimiter=',', dtype=np.float32)           

            # patience: 1
            # trainee : 0
            self.label[:, :,i] = 1 if (file.stem.split("_")[0] == 'patience') else 0

            proc.update()
        
    def __len__(self):
        return self.data.shape[2]
    
    def __getitem__(self, idx):
        # データ全部を渡す
        # シフトしながらDNNに入力するのは別の所で実装（だと思っている）
        return self.data[:,:,idx], self.label[:,:,idx]       


if __name__ == "__main__":

    batch_size = 10

    path = 'dataset/'

    dataset = MyDataset(path)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    for i, data in enumerate(dataloader):
        # shape: batch_size x siglen x ch
        inputs, labels = data

        print(inputs.shape)
        print(labels.shape)