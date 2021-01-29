import numpy as np
from tqdm import tqdm
import pickle
import cloudpickle

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 自作モジュール
from common.earlystopping import EarlyStopping
from datasetloader import MyDataset
from net import MyModel

writer = SummaryWriter()

# tensorboard 起動コマンド
# tensorboard --logdir runs/
# ssh ユーザ名@サーバーのIPアドレス -L 6006:localhost:6006

def train_step(epoch, dataloader, model, optimizer, train_i):

    model.train()
    for inputs, labels in tqdm(dataloader, desc=f'Training step[{epoch+1}/{MAX_EPOCH}]: ', leave=False):

        # tmp betch size データセットがうまく割り切れなかった時用
        tmp_batch_size = inputs.shape[0]

        # 任意の区間（frame_range）に信号をカットしDNNに入力
        # shift_sizeずつ区間をずらす
        for idx in range(0, siglen-frame_range, shift_size):

            # shape: batch_size x frame_range x ch
            input = inputs[:,idx:idx+frame_range,:]
            # conv2d用に次元追加
            # shape: batch_size x 1 x frame_range x ch
            input = input[:,np.newaxis,:,:]
            # データ数が少ないので適当にノイズ加算
            noise = np.random.normal(0, 0.01, input.shape).astype('float32')
            input = (input + noise).to(device)
            label = labels[:,idx:idx+frame_range,:].to(device)

            optimizer.zero_grad()

            # prediction
            output = model(input)

            # Loss計算
            try:
                # CrossEntropyLoss使用時
                loss = criterion(output, label[:,0,:].data.max(1)[1])
            except:
                # MSE使用時
                loss = criterion(output, label[:,0,:])

            loss.backward()    # バックプロパゲーション
            optimizer.step()   # 重み更新

            # 正解率計算　正解数/temp_batch_size
            # 0/1 に変換
            pred_label = output.data.max(1)[1] 
            accu_label = label[:,0,:].data.max(1)[1] 
            train_acc = torch.sum(pred_label==accu_label).cpu().numpy()/tmp_batch_size

            # tensorboard用に[loss, accu]を保存
            writer.add_scalar("Loss/Loss_train", loss,train_i)
            writer.add_scalar("Accuracy/Accu_train", train_acc,train_i)
            train_i+=1

def valid_step(epoch, dataloader, model, valid_i):

    valid_acc_sum, valid_acc_avg = 0, 0
    valid_loss_sum, valid_loss_avg = 0, 0

    itr = 0

    model.eval()
    for inputs, labels in tqdm(dataloader, desc=f'Validation step[{epoch+1}/{MAX_EPOCH}]: ', leave=False):
        # tmp betch size データセットがうまく割り切れなかった時用
        tmp_batch_size = inputs.shape[0]

        # 任意の区間（frame_range）に信号をカットしDNNに入力
        # shift_sizeずつ区間をずらす
        for idx in range(0, siglen-frame_range, shift_size):
            # shape: batch_size x frame_range x ch
            input = inputs[:,idx:idx+frame_range,:]

            # conv2d用に次元追加
            # shape: batch_size x 1 x frame_range x ch
            input = input[:,np.newaxis,:,:]

            # データ数が少ないので適当にノイズ加算
            noise = np.random.normal(0, 0.01, input.shape).astype('float32')
            input = (input + noise).to(device)
            label = labels[:,idx:idx+frame_range,:].to(device)

            with torch.no_grad():
                output = model(input)

            try:
                loss = criterion(output, label[:, 0, :].data.max(1)[1])
            except:
                loss = criterion(output, label[:, 0, :])
            
            valid_loss_sum += loss.item()

            # 0/1 に変換
            pred_label = output.data.max(1)[1]
            accu_label = label[:,0,:].data.max(1)[1]
            valid_acc  = torch.sum(pred_label==accu_label).cpu().numpy() / tmp_batch_size
            valid_acc_sum += valid_acc
            itr += 1
            # print(valid_acc)

            writer.add_scalar("Loss/Loss_valid", loss,valid_i) 
            writer.add_scalar("Accuracy/Accu_valid", valid_acc,valid_i)
            valid_i+=1

    valid_acc_avg = valid_acc_sum / itr
    valid_loss_avg = valid_loss_sum / itr

    # (1-Acc)とLossの和をスコアとしてEarlystoppingの指標とする
    # (結局Lossだけをみてあげれば良い気もする)
    return (1-valid_acc_avg) + valid_loss_avg

if __name__ == "__main__":

    shift_size  = 10
    frame_range = 3600
    
    MAX_EPOCH   = 2000
    BATCH_SIZE  = 10 # 1つのミニバッチのデータの数
    PATIENCE    = 20

    path = 'dataset/'

    try:
        # load
        with open('dataset/train/dataset.pickle', 'rb') as f:
            dataset = pickle.load(f)
    except:
        # make & save
        dataset = MyDataset(path)
        with open('dataset/train/dataset.pickle', 'wb') as f:
            pickle.dump(dataset,f)

    # 信号長を取得
    data, _ = dataset[0]
    siglen  = data.shape[0]

    # GPUが利用可能か確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # def model
    model = MyModel().to(device)
    
    early_stopping = EarlyStopping(PATIENCE, verbose=True, out_dir='dataset/models/', key='min')

    # initial setting
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    train_i, valid_i = 0, 0

    # Dataset =====================================================
    n_samples = len(dataset) # n_samples is 200 files
    train_size = int(len(dataset) * 0.9) # train_size is 180 files
    val_size = n_samples - train_size # val_size is 20 files

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(val_dataset,   BATCH_SIZE, shuffle=True)
    # 高速化
    # train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    # valid_dataloader = DataLoader(val_dataset,   BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # =============================================================

    # DNN Training ================================================
    for epoch in range(MAX_EPOCH):
        train_step(epoch, train_dataloader, model, optimizer, train_i)
        valid_score = valid_step(epoch, valid_dataloader, model,valid_i)

        early_stopping(valid_score, model)

        if early_stopping.early_stop:
            print('Early stopping.')
            break
    # =============================================================

    # 学習済みモデルの保存
    with open('dataset/models/trained_model.pickle', 'wb') as f:
        cloudpickle.dump(model, f)