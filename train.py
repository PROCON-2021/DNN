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
from datasetloader import TrainValDataset
from common.functions import check_loss

os.environ["WANDB_SILENT"] = "true"

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

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # pred_label = output.data.max(1)[1] 
        # accu_label = labels[:,0,:].data.max(1)[1] 
        # train_acc = torch.sum(pred_label==accu_label).cpu().numpy()/tmp_batch_size

        if (i+1) % check_interval == 0:
            # lossのデバッグ
            check_loss(epoch, i, check_interval, running_loss)
            wandb.log({'Training loss': running_loss})
            running_loss = 0.0

            # accのデバッグ
            correct = 0
            _, predicted = t.max(output.data, 1)
            total = labels.size(0)
            correct += (predicted == labels).sum().item()
            wandb.log({'Training accuracy': correct / total})

def valid_step(dataloader, model):

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
    loss_avg = valid_loss_sum / (i+1)
    wandb.log({'Validation loss': loss_avg})

    return acc

if __name__ == "__main__":

    type_ = 'abs'
    boost = True

    MAX_EPOCH   = 2000
    BATCH_SIZE  = 10
    PATIENCE    = 20

    wandb.init(project=type_)

    out_dir = wandb.run.dir
    path = Path(out_dir)
    out_dir = '/'.join(path.parts[0:-1])

    # GPUが利用可能か確認
    device = 'cuda' if t.cuda.is_available() else 'cpu'

    # Dataset =====================================================
    train_dataset = TrainValDataset(f'./dataset/{type_}/train', type_)
    val_dataset = TrainValDataset(f'./dataset/{type_}/val', type_)

    if boost == True:
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    out_dim = len(train_dataset.label)
    model = MyModel(out_dim=out_dim).to(device)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, out_dir=out_dir, key='max')

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DNN Training ================================================
    for epoch in range(MAX_EPOCH):
        train_step(epoch, train_loader, model, optimizer)
        acc = valid_step(valid_loader, model)

        wandb.log({'Validation accuracy': acc, 'Epoch': epoch+1})
        early_stopping(acc, model)

        if early_stopping.early_stop:
            print('Early stopping.')
            break
    # =============================================================