import numpy as np
import torch.nn as nn
import torch as t

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_out_size(self, in_size, kernel_size, stride, padding):
        return int(np.floor(((in_size + (2*padding) - (kernel_size-1) - 1) // stride) + 1))

    def forward(self, x):

        x = self.conv(x)  # conv.
        x = t.flatten(x, 1)  # flatten
        x = self.dense(x)  # dense

        return x

class Conv1dModel(ConvModel):
    def __init__(self, out_dim, p=0):
        super().__init__()

        input_size  = 448
        hidden1     = 256
        hidden2     = 128
        hidden3     = 64
        hidden4     = 32

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=5, stride=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden2, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(hidden3, out_dim),
        )

class Conv2dModel(ConvModel):
    def __init__(self, out_dim, p=0):
        super().__init__()

        input_size  = 2240
        hidden1     = 512
        hidden2     = 512

        out_channels = [16, 32, 64, 64]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels[0], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[0]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[1]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[2]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=[2, 2], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[3]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=[2, 2], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[3]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Dropout2d(p=0.5),
        )

        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden2, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden2, out_dim),
        )