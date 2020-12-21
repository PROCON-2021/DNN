import numpy as np
from torch.utils.data import Dataset, DataLoader

FRAME_RANGE = 20

class MyDataset(Dataset):
    def __init__(self, sig):
        self.sig = sig
        pass

    def __len__(self):
        return self.sig.shape[0]
    
    def __getitem__(self, idx):
        label = np.ones([FRAME_RANGE, 4])
        inputs = np.zeros([FRAME_RANGE, self.sig.shape[1]])

        # idxが範囲外を指すとき
        if idx+FRAME_RANGE >= self.__len__():
            inputs[0:self.__len__()-idx, :] = sig[idx:self.__len__(), :]
            inputs[self.__len__()-idx-1, :] = 0
        else:
            inputs = sig[idx:idx+FRAME_RANGE, :]

        return inputs, label

if __name__ == "__main__":

    batch_size = 10

    sig = np.loadtxt('sig.npy', dtype=np.float32, delimiter=',')

    dataset = MyDataset(sig)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    for i, data in enumerate(dataloader):
        inputs, labels = data
        print(inputs.shape)
        print(labels.shape)