import torch.nn as nn
import torch as t

class MyModel(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        input_size  = 2604
        hidden1     = 1024*4
        hidden2     = 1024*1
        hidden3     = 512
        hidden4     = 64

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=[3, 3], stride=[2, 1], padding=[0, 1]),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[3, 3], stride=[2, 1], padding=[0, 1]),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[2, 1], padding=[0, 1]),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=[3, 3], stride=[2, 1], padding=[0, 1]),
            nn.LeakyReLU(),
        )

        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden1),  nn.BatchNorm1d(hidden1),     nn.LeakyReLU(0.2, inplace=True), nn.Dropout(p=0.3),
            nn.Linear(hidden1, hidden2),     nn.BatchNorm1d(hidden2),     nn.LeakyReLU(0.2, inplace=True), nn.Dropout(p=0.3),
            nn.Linear(hidden2, hidden2),     nn.BatchNorm1d(hidden2),     nn.LeakyReLU(0.2, inplace=True), nn.Dropout(p=0.3),
            nn.Linear(hidden2, hidden3),     nn.BatchNorm1d(hidden3),     nn.LeakyReLU(0.2, inplace=True), nn.Dropout(p=0.2),
            nn.Linear(hidden3, hidden4),     nn.BatchNorm1d(hidden4),     nn.LeakyReLU(0.2, inplace=True), nn.Dropout(p=0.2),
            nn.Linear(hidden4, out_dim), nn.BatchNorm1d(out_dim), nn.LeakyReLU(0.2, inplace=True)
        )


    def forward(self, x):

        x = self.conv(x)  # conv.
        x = t.flatten(x, 1)  # flatten
        x = self.dense(x)  # dense

        return x