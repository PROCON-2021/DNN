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
    def __init__(self, h, w, channels, n_layers, conv_kernel, conv_stride, pool_kernel, pool_stride, h_dims, out_dim, p=0):
        super().__init__()

        self.h_out = 0
        self.w_out = 0

        # 第1層
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=[conv_kernel,3], stride=[conv_stride,1], padding=[0,1]),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=[pool_kernel, 1], stride=[pool_stride, 1], padding=0),
            )
        ])

        h_ = self.calc_out_size(h, conv_kernel, conv_stride, 0)
        h_ = self.calc_out_size(h_, pool_kernel, pool_stride, 0)

        w_ = self.calc_out_size(w, 3, 1, 1)
        w_ = self.calc_out_size(w_, 1, 1, 0)

        # 2層目以降．
        for i in range(n_layers-1):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=[conv_kernel,3], stride=[conv_stride,1], padding=[0,1]),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=[pool_kernel, 1], stride=[pool_stride, 1], padding=0),
                )
            )

            h_ = self.calc_out_size(h_, conv_kernel, conv_stride, 0)
            h_ = self.calc_out_size(h_, pool_kernel, pool_stride, 0)
            w_ = self.calc_out_size(w_, 3, 1, 1)
            w_ = self.calc_out_size(w_, 1, 1, 0)
        
        self.h_out = h_
        self.w_out = w_

        out_features = channels[n_layers-1]*self.h_out*self.w_out

        self.dense = nn.Sequential(
            nn.Linear(out_features, h_dims[0]),
            nn.BatchNorm1d(h_dims[0]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(h_dims[0], h_dims[1]),
            nn.BatchNorm1d(h_dims[1]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(h_dims[1], h_dims[2]),
            nn.BatchNorm1d(h_dims[2]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(h_dims[2], h_dims[3]),
            nn.BatchNorm1d(h_dims[3]),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(h_dims[3], out_dim),
        )
    def forward(self, x):

        # conv.
        for l in self.conv:
            x = l(x)
        
        x = t.flatten(x, 1)  # flatten
        x = self.dense(x)  # dense

        return x