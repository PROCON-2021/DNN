from os import write
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard.summary import hparams
from tqdm import tqdm
import datetime
import pandas
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle
from torch.utils.tensorboard import SummaryWriter
import cloudpickle
import optuna
from common.earlystopping import EarlyStopping
from datasetloader import MyDataset

#tensorboard 機動コマンド
#tensorboard --logdir runs/

class MyModel(nn.Module):
    def __init__(self, p_ratio=0):

        super(MyModel, self).__init__()

        self.conv_layer1 = nn.Sequential(
        nn.Conv2d(in_channels =1, out_channels = 4, kernel_size =[3, 3], stride =[2, 1],padding =[0, 1]),
        nn.LeakyReLU())
        self.conv_layer2 = nn.Sequential(
        nn.Conv2d(in_channels =4, out_channels = 16, kernel_size =[3, 3], stride =[2, 1],padding =[0, 1]),
        nn.LeakyReLU())
        self.conv_layer3 = nn.Sequential(
        nn.Conv2d(in_channels =16, out_channels = 16, kernel_size =[3, 3], stride =[2, 1],padding =[0, 1]),
        nn.LeakyReLU())
        self.conv_layer4 = nn.Sequential(
        nn.Conv2d(in_channels =16, out_channels = 4, kernel_size =[3, 3], stride =[2, 1],padding =[0, 1]),
        nn.LeakyReLU())


        input_size = 2688
        hidden1 = 1024*4
        hidden2 = 1024*1
        hidden3 = 512
        hidden4 = 64
        output_size = 2

        self.nural = nn.Sequential(
            nn.Linear(input_size,hidden1),nn.BatchNorm1d(hidden1), nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=p_ratio),
            nn.Linear(hidden1,hidden2),nn.BatchNorm1d(hidden2),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=p_ratio),
            nn.Linear(hidden2,hidden2),nn.BatchNorm1d(hidden2),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=p_ratio),
            nn.Linear(hidden2,hidden3),nn.BatchNorm1d(hidden3),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.2),
            nn.Linear(hidden3,hidden4),nn.BatchNorm1d(hidden4),nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.2),
            nn.Linear(hidden4,output_size),nn.BatchNorm1d(output_size),nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = self.conv_layer4(output)

        self.output = output.view(-1, self.num_flat_features(output))#flatten
        self.output = self.nural(self.output)
        self.output = self.output.view(-1,2)
        self.output = F.softmax(self.output, dim = -1)
        return self.output                                                     

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def objective(trial):

    dir = ['runs', date, str(uuid.uuid4())]
    writer = SummaryWriter(log_dir=('/').join(dir))

    # ハイパーパラメータ
    batch_size = trial.suggest_int('batch_size', 16, 32)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2)

    shift_size  = 10
    frame_range = 3600
    
    MAX_EPOCH   = 10
    PATIENCE    = 20

    path = 'dataset/'

    try:
        #load
        with open('dataset/train/dataset.pickle', 'rb') as f:
            dataset = pickle.load(f)
    except:
        #make & save
        dataset = MyDataset(path)
        with open('dataset/train/dataset.pickle', 'wb') as f:
            pickle.dump(dataset,f)

    # 信号長を取得
    data, _ = dataset[0]
    siglen = data.shape[0]

    # GPUが利用可能か確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    # def model
    model = MyModel(p_ratio=dropout)
    model = model.to(device)
    
    early_stopping = EarlyStopping(PATIENCE, verbose=True, out_dir='dataset/models/', key='max')

    # initial setting
    # learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    train_i,valid_i = 0,0

    n_samples = len(dataset) # n_samples is 200 files
    train_size = int(len(dataset) * 0.9) # train_size is 180 files
    val_size = n_samples - train_size # val_size is 20 files

    #dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_dataset,   batch_size, shuffle=True)

    # OptunaチューニングとEarlystoppingの為に計算
    valid_acc_sum = 0
    valid_acc_ave = 0

    hparams = {
        'batch_size': batch_size,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'shift_size': shift_size,
        'frame_range': frame_range,
    }

    for epoc in range(MAX_EPOCH):
        #######################################################
        #                     Train mode
        #######################################################
        for data, label in tqdm(train_dataloader):
        #for i, data in enumerate(train_dataloader):

            # shape: batch_size x siglen x ch
            inputs, labels = data, label
            # tmp betch size データセットがうまく割り切れなかった時用
            tmp_batch_size = data.shape[0]

            # 任意の区間（frame_range）に信号をカットしDNNに入力
            # shift_sizeずつ区間をずらす
            for idx in range(0, siglen-frame_range, shift_size):

                # shape: batch_size x frame_range x ch
                input = inputs[:,idx:idx+frame_range,:]
                # conv2d用に次元追加　　shape: batch_size x 1 x frame_range x ch
                input = input[:,np.newaxis,:,:]
                #データ数が少ないので適当にノイズ加算
                noise = np.random.normal(0, 0.01, input.shape).astype('float32')
                input = (input + noise).to(device)
                label = labels[:,idx:idx+frame_range,:].to(device)

                optimizer.zero_grad()

                #prediction
                output = model(input)

                #Loss計算
                try:#nn.CrossEntropyLoss()使用時
                    loss = criterion(output, label[:,0,:].data.max(1)[1])
                except:# MSE使用時
                    loss = criterion(output, label[:,0,:])

                loss.backward()    #バックプロパゲーション
                optimizer.step()   # 重み更新   .

                #正解率計算　正解数/temp_batch_size
                pred_label = output.data.max(1)[1] #予測結果を01に変換
                accu_label = label[:,0,:].data.max(1)[1] #正解を01に変換
                train_acc = torch.sum(pred_label==accu_label).cpu().numpy()/tmp_batch_size
                #print(train_acc)

                #tensorboard用に [loss, accu]を保存
                writer.add_scalar("Loss/Loss_train", loss,train_i)#log loss
                writer.add_scalar("Accuracy/Accu_train", train_acc,train_i)#log loss
                train_i+=1

        #######################################################
        #                     Validation mode
        #######################################################

        # 平均のvalid_accを算出するためにイタレーションを数える
        itr = 0

        for data, label in tqdm(valid_dataloader):

            # shape: batch_size x siglen x ch
            inputs, labels = data, label
            # tmp betch size データセットがうまく割り切れなかった時用
            tmp_batch_size = data.shape[0]

            # 任意の区間（frame_range）に信号をカットしDNNに入力
            # shift_sizeずつ区間をずらす
            for idx in range(0, siglen-frame_range, shift_size):

                # shape: batch_size x frame_range x ch
                input = inputs[:,idx:idx+frame_range,:]
                # conv2d用に次元追加　　shape: batch_size x 1 x frame_range x ch
                input = input[:,np.newaxis,:,:]
                #データ数が少ないので適当にノイズ加算
                noise = np.random.normal(0, 0.01, input.shape).astype('float32')
                input = (input + noise).to(device)
                label = labels[:,idx:idx+frame_range,:].to(device)

                optimizer.zero_grad()

                #prediction
                with torch.no_grad():
                    output = model(input)
               
                try:
                    loss = criterion(output, label[:, 0, :].data.max(1)[1])
                except:
                    loss = criterion(output, label[:, 0, :])

                pred_label = output.data.max(1)[1] #予測結果を01に変換
                accu_label = label[:,0,:].data.max(1)[1] #正解を01に変換
                valid_acc = torch.sum(pred_label==accu_label).cpu().numpy()/tmp_batch_size
                valid_acc_sum += valid_acc
                # print(valid_acc)

                writer.add_scalar("Loss/Loss_valid", loss,valid_i)#log loss
                writer.add_scalar("Accuracy/Accu_valid", valid_acc,valid_i)#log loss
                valid_i+=1
                itr += 1

        valid_acc_ave = valid_acc_sum / itr
        # print(f'Average valid_acc:{valid_acc_ave}')

        early_stopping(valid_acc_ave, model)
                
        # Earlystopping
        if early_stopping.early_stop:
            print('Early stopping.')
            break
    
        # Optunaの枝刈り
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        valid_acc_sum = valid_acc_ave = 0

    writer.add_hparams(hparams, {"hparam/average_valid_acc": valid_acc_ave})
    writer.close()

    return valid_acc_ave

    # #学習済みモデルの保存
    # with open('dataset/models/trained_model.pickle', 'wb') as f:
    #     cloudpickle.dump(model, f)

if __name__ == "__main__":

    date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Optuna 枝刈り手法
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=4,
        min_early_stopping_rate=0
    )

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=5)

    study_df = study.trials_dataframe()
    study_df.to_csv(date+'result.csv')

#tensorboard 機動コマンド

#ssh ユーザ名@サーバーのIPアドレス -L 6006:localhost:6006
#tensorboard --logdir runs/