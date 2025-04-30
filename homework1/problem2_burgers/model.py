import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet2D(nn.Module):

    # You may include additional arguments if you wish.
    def __init__(self):
        # TODO
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Tanh1 = nn.Tanh()

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Tanh2 = nn.Tanh()

        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Tanh3 = nn.Tanh()

        self.conv4 = nn.Conv2d(64, 128, 1)
        self.Tanh4 = nn.Tanh()
        self.fc2 = nn.Conv2d(128, 1, kernel_size=1) #Check during OH

    def forward(self, x):
        # TODO
        x = x.permute(0, 3, 1, 2)
        x = torch.tanh(self.conv1(x))
        x = self.Tanh2(self.conv2(x))
        x = self.Tanh3(self.conv3(x))
        x = self.Tanh4(self.conv4(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x
