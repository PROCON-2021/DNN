import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        input_size  = 2688
        hidden1     = 1024*4
        hidden2     = 1024*1
        hidden3     = 512
        hidden4     = 64
        output_size = 2

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
            nn.Linear(hidden4, output_size), nn.BatchNorm1d(output_size), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):

        # conv.
        output = self.conv(x)

        # Flatten
        output = output.view(-1, self.num_flat_features(output))

        # dense
        output = self.dense(output)
        output = output.view(-1,2)
        output = F.softmax(output, dim = -1)
        return output                                                     

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features