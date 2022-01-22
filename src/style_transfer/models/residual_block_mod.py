from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self,in_out_channel):
        """
        :param in_out_channel: number of input and output channel
        """
        super(ResidualBlock,self).__init__()
        self.direct_path = nn.Sequential(
            nn.Conv2d(in_out_channel,in_out_channel,kernel_size=(3,3),padding = "same"),
            nn.BatchNorm2d(in_out_channel),
            nn.ReLU(),
            nn.Conv2d(in_out_channel,in_out_channel,kernel_size=(3,3),padding = "same"),
            nn.BatchNorm2d(in_out_channel))
    def forward(self,inpt):
        out = inpt + self.direct_path(inpt)
        return out