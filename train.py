#-----------------------------------------
# import
#-----------------------------------------
import os
import numpy as np
#import cloudpickle
import sys
#import soundfile as sf
#import pylab as plt
#import wave
#import struct
from scipy import fromstring, int16,signal
import random
#import pickle
#import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import csv
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
from tqdm import tqdm

#print(torch.__version__) # 0.4.0
#tensorboard 機動コマンド
#tensorboard --logdir runs/

#-----------------------------------------
# functions
#-----------------------------------------
class DataSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        self.label = []
        
        for _ in range(200):
            input_data = torch.rand([3,100], dtype= torch.float)
            target     = torch.eye(2)[0]
            self.data.append(input_data)
            self.label.append(target)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]
        return out_data, out_label

#-----------------------------------------
# DNN model
#-----------------------------------------
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        input_size = 300
        hidden1 = 1024*2
        hidden2 = 1024*1
        hidden3 = 512
        hidden4 = 64
        output_size = 2

        self.nural = nn.Sequential(
            nn.Linear(input_size,hidden1),nn.BatchNorm1d(hidden1), nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.3),
            nn.Linear(hidden1,hidden2),nn.BatchNorm1d(hidden2),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.3),
            nn.Linear(hidden2,hidden2),nn.BatchNorm1d(hidden2),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.3),
            nn.Linear(hidden2,hidden3),nn.BatchNorm1d(hidden3),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.2),
            nn.Linear(hidden3,hidden4),nn.BatchNorm1d(hidden4),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.2),
            nn.Linear(hidden4,output_size),nn.BatchNorm1d(output_size),nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        output_flat = x.view(-1, self.num_flat_features(x))#flatten
        self.output = self.nural(output_flat)
        self.output = self.output.view(-1,2)
        self.output = F.softmax(self.output, dim = -1)
        return self.output

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
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # Load model
    model = MyModel()
    model = model.to(device)

    # initial setting
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()

    loss_log = [] # 学習状況のプロット用
    acc_log = [] # 学習状況のプロット用
    train_loss = []
    train_accu = []
    train_i,val_i = 0,0
    BATCH_SIZE  = 128 # 1つのミニバッチのデータの数
    
    train_data_set = DataSet()
    valid_data_set = DataSet()
    
    train_size = (len(train_data_set)//BATCH_SIZE)*BATCH_SIZE
    train_data_set, dust = torch.utils.data.random_split(train_data_set, [train_size, len(train_data_set)-train_size])
    valid_size = (len(valid_data_set)//BATCH_SIZE)*BATCH_SIZE
    valid_data_set, dust = torch.utils.data.random_split(valid_data_set, [valid_size, len(valid_data_set)-valid_size])


    train_loader = torch.utils.data.DataLoader(
        dataset=train_data_set,  # データセットの指定
        batch_size=BATCH_SIZE,  # ミニバッチの指定
        shuffle=True,  # シャッフルするかどうかの指定
        num_workers=0)  # コアの数
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0)
    print('train_dataset = ', len(train_data_set))
    print('valid_dataset = ', len(valid_data_set))

    #training step
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    train_loss, train_acc, val_loss, val_acc,t = 0, 0, 0, 0,0

    for epoc in range(2000):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        # ======== train_mode ======
        model.train() #学習モード
        for in_data, labels in tqdm(train_loader): # 1ミニバッチずつ計算
            in_data, labels = Variable(in_data), Variable(labels)#微分可能な型
            in_data = in_data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            #Prediction
            pred_y = model(in_data)

            train_loss = criterion(pred_y, labels)
            train_loss.backward()    #バックプロパゲーション
            optimizer.step()   # 重み更新

            accu_label = torch.argmax(labels, dim = -1)
            pred_label = torch.argmax(pred_y, dim = -1)
            train_accu = torch.sum(pred_label==accu_label) / BATCH_SIZE

            #writer.add_scalar("Loss/Loss_train", loss,train_i)#log train_loss
            #writer.add_scalar("Accuracy/Accu_train", train_acc,train_i)#log train_accu
            train_i+=1

        train_loss_list.append(train_loss.item())
        train_acc_list.append(train_accu)


        # ======== valid_mode ======
        model.eval() #学習モード
        for in_data, labels in tqdm(valid_loader): # 1ミニバッチずつ計算

            in_data, labels = Variable(in_data), Variable(labels)#微分可能な型
            in_data = in_data.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                
                #Prediction
                pred_y = model(in_data)

                val_loss = criterion(pred_y, labels)
                accu_label = torch.argmax(labels, dim = -1)
                pred_label = torch.argmax(pred_y, dim = -1)
                val_accu = torch.sum(pred_label==accu_label) / BATCH_SIZE

                #writer.add_scalar("Loss/Loss_val", loss,val_i)#log val_loss
                #writer.add_scalar("Accuracy/Accu_val", val_acc,val_i)#log val_accu
                val_i+=1
        #writer.close()
        val_loss_list.append(val_loss.item())
        val_acc_list.append(val_accu)

        print("Epoc:{}, val_accu:{}, val_loss:{}".format(epoc, val_accu, val_loss))
        
        if val_acc>=0.970:
            break
    #with open('./Output/trained_model/model_est_subband.pkl', 'wb') as f:
    #    cloudpickle.dump(model, f)
    
    
#tensorboard 機動コマンド
#tensorboard --logdir runs/











