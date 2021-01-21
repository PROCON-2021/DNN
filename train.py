#-----------------------------------------
# import
#-----------------------------------------
import os, sys
import numpy as np
from tqdm import tqdm

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasetloader import MyDataset

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

#print(torch.__version__) # 0.4.0
#tensorboard 機動com
#tensorboard --logdir runs/

#-----------------------------------------
# DNN model
#-----------------------------------------
class MyModel(nn.Module):
    def __init__(self, frame_range, ch=4):
        super(MyModel, self).__init__()
        input_size = frame_range*ch
        hidden1 = 512
        hidden2 = 512
        hidden3 = 256
        hidden4 = 256
        output_size = frame_range

        self.frame_range = frame_range

        self.nural = nn.Sequential(
            nn.Linear(input_size,hidden1),nn.BatchNorm1d(hidden1), nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.3),
            nn.Linear(hidden1,hidden2),nn.BatchNorm1d(hidden2),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.3),
            nn.Linear(hidden2,hidden2),nn.BatchNorm1d(hidden2),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.3),
            nn.Linear(hidden2,hidden3),nn.BatchNorm1d(hidden3),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.2),
            nn.Linear(hidden3,hidden4),nn.BatchNorm1d(hidden4),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.2),
            nn.Linear(hidden4,output_size),nn.BatchNorm1d(output_size),nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        output_flat = x.view(-1, self.num_flat_features(x))  #flatten
        output = self.nural(output_flat)
        output = output.view(-1, self.frame_range, 1)
        output = F.softmax(output, dim=-1)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#-----------------------------------------
# main
#-----------------------------------------
if __name__ == "__main__":
    # GPUが利用可能か確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_log = [] # 学習状況のプロット用
    acc_log = [] # 学習状況のプロット用
    train_loss = []
    train_accu = []
    train_i,val_i = 0,0

    MAX_EPOCH   = 2000
    BATCH_SIZE  = 10 # 1つのミニバッチのデータの数

    dataset = MyDataset('dataset/')

    # データセットをTrainとValに分割 
    samples = len(dataset)
    train_size = int(samples*0.8)
    val_size = samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,  # データセットの指定
        batch_size=BATCH_SIZE,  # ミニバッチの指定
        shuffle=True,  # シャッフルするかどうかの指定
        num_workers=0)  # コアの数
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0)

    print('train_dataset = ', len(train_dataset))
    print('val_dataset = ', len(val_dataset))

    #training step
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    train_loss, train_acc, val_loss, val_acc,t = 0, 0, 0, 0,0

    data, _ = dataset[0]
    siglen = data.shape[0]

    frame_range = 1000
    shift_size = 200

    # Load model
    model = MyModel(frame_range, ch=4)
    model = model.to(device)

    # initial setting
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    for epoc in range(MAX_EPOCH):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        # ======== train_mode ======
        model.train() #学習モード
        for in_data, labels in tqdm(train_loader): # 1ミニバッチずつ計算
            in_data = in_data.to(device)
            labels  = labels.to(device)

            # 任意の区間（frame_range）に信号をカットしDNNに入力
            # shift_sizeずつ区間をずらす
            for idx in range(0, siglen-frame_range, shift_size):
                optimizer.zero_grad()

                # shape: batch_size x frame_range x ch
                input = in_data[:,idx:idx+frame_range,:]
                label = labels[:,idx:idx+frame_range,:]

                #Prediction
                pred_y = model(input)

                train_loss = criterion(pred_y, label)
                train_loss.backward()    #バックプロパゲーション
                optimizer.step()   # 重み更新

                # acc_label = torch.argmax(label, dim = -1)
                # pred_label = torch.argmax(pred_y, dim = -1)
                # train_acc = torch.sum(pred_label==acc_label) / ((siglen-frame_range) /shift_size)

            train_loss_list.append(train_loss.item())
            # train_acc_list.append(train_acc)

            #writer.add_scalar("Loss/Loss_train", loss,train_i)#log train_loss
            #writer.add_scalar("Accuracy/Accu_train", train_acc,train_i)#log train_accu
            train_i+=1

        # ======== val_mode ======
        model.eval() #学習モード
        for in_data, labels in tqdm(val_loader): # 1ミニバッチずつ計算

            in_data = in_data.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                # 任意の区間（frame_range）に信号をカットしDNNに入力
                # shift_sizeずつ区間をずらす
                for idx in range(0, siglen-frame_range, shift_size):

                    # shape: batch_size x frame_range x ch
                    input = in_data[:,idx:idx+frame_range,:]
                    label = labels[:,idx:idx+frame_range,:]

                    #Prediction
                    pred_y = model(input)

                    val_loss = criterion(pred_y, label)

                    acc_label = torch.argmax(label, dim = -1)
                    pred_label = torch.argmax(pred_y, dim = -1)
                    val_acc = torch.sum(pred_label==acc_label) / siglen

                    #writer.add_scalar("Loss/Loss_val", loss,val_i)#log val_loss
                    #writer.add_scalar("Accuracy/Accu_val", val_acc,val_i)#log val_acc
                val_i+=1

            val_loss_list.append(val_loss.item())
            val_acc_list.append(val_acc)

        #writer.close()
        print("Epoc:{}, val_acc:{}, val_loss:{}".format(epoc, val_acc, val_loss))

        if val_acc>=0.970:
            break
    #with open('./Output/trained_model/model_est_subband.pkl', 'wb') as f:
    #    cloudpickle.dump(model, f)

#tensorboard 機動コマンド
#tensorboard --logdir runs/