import os
import numpy as np
import wandb
from pathlib import Path

# PyTorch
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 自作モジュール
from common.earlystopping import EarlyStopping
from net import MyModel
from datasetloader import ThighsTrainValDataset
from common.functions import check_loss

os.environ["WANDB_SILENT"] = "true"
PROJECT_NAME = 'Thighs'

def train_step(epoch, dataloader, model, optimizer):

    running_loss = 0.0
    check_interval = 10

    model.train()
    for i, data in enumerate(dataloader):

        inputs, labels = [i.to(device) for i in data]

        optimizer.zero_grad()

        # prediction
        output = model(inputs)

        # CrossEntropyLoss
        loss = criterion(output, labels)

        loss.backward()    # バックプロパゲーション
        optimizer.step()   # 重み更新

        running_loss += loss.item()

        # pred_label = output.data.max(1)[1] 
        # accu_label = labels[:,0,:].data.max(1)[1] 
        # train_acc = torch.sum(pred_label==accu_label).cpu().numpy()/tmp_batch_size

        if (i+1) % check_interval == 0:
            check_loss(epoch, i, check_interval, running_loss)
            wandb.log({'Training loss': running_loss})
            running_loss = 0.0

def valid_step(epoch, dataloader, model):

    valid_loss_sum = 0

    correct = 0
    total = 0

    model.eval()
    with t.no_grad():
        for i, data in enumerate(dataloader):

            inputs, labels = [i.to(device) for i in data]
    
            output = model(inputs)

            loss = criterion(output, labels)

            valid_loss_sum += loss.item()

            _, predicted = t.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (correct / total)
    return acc

if __name__ == "__main__":

    MAX_EPOCH   = 2000
    BATCH_SIZE  = 10 # 1つのミニバッチのデータの数
    PATIENCE    = 20

    classes = 6

    wandb.init(project=PROJECT_NAME)

    out_dir = wandb.run.dir
    path = Path(out_dir)
    out_dir = '/'.join(path.parts[0:-1])

    # GPUが利用可能か確認
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    model = MyModel(out_dim=6).to(device)  # Model
    early_stopping = EarlyStopping(PATIENCE, verbose=True, out_dir=out_dir, key='max')

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Dataset =====================================================
    train_dataset = ThighsTrainValDataset('./dataset/thighs/train')
    val_dataset = ThighsTrainValDataset('./dataset/thighs/val')

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    # train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    # valid_loader = DataLoader(val_dataset,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # DNN Training ================================================
    for epoch in range(MAX_EPOCH):
        train_step(epoch, train_loader, model, optimizer)
        acc = valid_step(epoch, valid_loader, model)

        wandb.log({'Validation accuracy': acc, 'Epoch': epoch+1})
        early_stopping(acc, model)

        if early_stopping.early_stop:
            print('Early stopping.')
            break
    # =============================================================