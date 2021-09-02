import os
import numpy as np
import wandb
from pathlib import Path
import optuna
import argparse

# PyTorch
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
t.manual_seed(0)

# 自作モジュール
from common.earlystopping import EarlyStopping
from net import Conv2dModel, Conv1dModel
from datasetloader import TrainValDataset
from common.functions import export_network, get_optimzer

os.environ["WANDB_SILENT"] = "true"

def train_step(epoch, dataloader, model, optimizer, run):

    running_loss = 0.0
    check_interval = 5
    total = 0
    correct = 0

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

        # Acc計算
        total += labels.size(0)
        _, predicted = t.max(output.data, 1)
        correct += (predicted == labels).sum().item()

        if (i+1) % check_interval == 0:
            print(f'[{epoch+1}, {i+1:03}] loss: {running_loss/check_interval:.5f}, Acc: {correct/total:.5f}')

            run.log({'Training loss': running_loss/check_interval})
            run.log({'Training accuracy': correct / total})

            running_loss = 0.0
            total = correct = 0

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

def objective(trial):

    # ハイパーパラメータ
    batch_size = trial.suggest_int('batch_size', 10, 50)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'RAdam'])
    norm = trial.suggest_categorical("norm", [True, False])
    lr = trial.suggest_float('lr', 1e-5, 1e-2)

    o_channel1 = trial.suggest_int('o_channel1', 2, 8)
    o_channel2 = trial.suggest_int('o_channel2', o_channel1, 16)
    o_channel3 = trial.suggest_int('o_channel3', o_channel2, 16)
    o_channel4 = trial.suggest_int('o_channel4', o_channel3, 16)

    h_dim1 = trial.suggest_int('h_dim1', 512, 1024)
    h_dim2 = trial.suggest_int('h_dim2', 256, 512)
    h_dim3 = trial.suggest_int('h_dim3', 128, 256)
    h_dim4 = trial.suggest_int('h_dim4', 32, 128)

    o_channels = [o_channel1, o_channel2, o_channel3, o_channel4]
    h_dims = [h_dim1, h_dim2, h_dim3, h_dim4]

    layers = trial.suggest_int('layers', 2, len(o_channels))

    conv_kernel = trial.suggest_int('conv_kernel', 1, 3)
    conv_stride = trial.suggest_int('conv_stride', 1, 2)

    pool_kernel = trial.suggest_int('pool_kernel', 1, 4)
    pool_stride = trial.suggest_int('pool_stride', 1, 4)

    config = dict(
        batch = batch_size,
        dropout = dropout,
        optimizer = optimizer_name,
        learning_rate = lr,
        best_val_acc = 0,
        pruned = False,
        conv_channels = o_channels,
        hidden_dims = h_dims,
        layers = layers,
        conv_kernel = conv_kernel,
        conv_stride = conv_stride,
        pool_kernel = pool_kernel,
        pool_stride = pool_stride,
        norm = norm,
    )

    run = wandb.init(project=args.type, config=config, reinit=True)

    out_dir = wandb.run.dir
    path = Path(out_dir)
    out_dir = '/'.join(path.parts[0:-1])

    # Dataset =====================================================
    train_dataset = TrainValDataset(f'./dataset/{args.type}/train', args.type, norm)
    val_dataset = TrainValDataset(f'./dataset/{args.type}/val', args.type, norm)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=args.workers)

    data, _ = next(iter(train_loader))

    if data.ndim == 4:
        *_, h, w = data.shape
    # Conv1D用
    elif data.ndim == 3:
        _, c, l = data.shape

    out_dim = len(train_dataset.label)
    # model = Conv1dModel(out_dim=out_dim, p=dropout).to(device)
    model = Conv2dModel(h, w, o_channels, layers, conv_kernel, conv_stride, pool_kernel, pool_stride, h_dims, out_dim, p=dropout).to(device)

    # DEBUG
    # summary(model, (1, h, w))
    
    early_stopping = EarlyStopping(args.patience, verbose=True, out_dir=out_dir, key='max')

    optimizer = get_optimzer(optimizer_name, model, lr)

    acc = 0
    best_val_acc = 0

    # DNN Training ================================================
    for epoch in range(MAX_EPOCH):
        train_step(epoch, train_loader, model, optimizer, run)
        acc = valid_step(valid_loader, model)

        run.log({'Validation accuracy': acc, 'Epoch': epoch+1})
        run.config.update({'best_val_acc': acc if acc > best_val_acc else best_val_acc}, allow_val_change=True)

        early_stopping(acc, model)

        trial.report(acc, epoch)

        if early_stopping.early_stop:
            print('Early stopping.')
            break

        if trial.should_prune():
            run.config.update({'pruned': True}, allow_val_change=True)
            export_network(model, wandb.run.dir)
            raise optuna.exceptions.TrialPruned()
    # =============================================================
    export_network(model, wandb.run.dir)
    run.finish()

    return acc

if __name__ == "__main__":

    MAX_EPOCH = 2000

    # GPUが利用可能か確認
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print('Device:', device)

    parser = argparse.ArgumentParser()

    parser.add_argument("type", type=str, help="筋トレ種目")
    parser.add_argument("--workers", "-w", type=int, default=0, help="Number of num_workers")
    parser.add_argument("--patience", "-p", type=int, default=20, help="patience epochs for early stopping")
    parser.add_argument("--trial",type=int, default=100)

    args = parser.parse_args()

    criterion = nn.CrossEntropyLoss()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10,
        interval_steps=1,
    )

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=args.trial)