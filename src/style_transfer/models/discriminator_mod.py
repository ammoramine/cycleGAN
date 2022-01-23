from torch import nn


class MainBlock(nn.Module):
    def __init__(self,in_channel,out_channel,use_instance_norm=True):
        super(MainBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_instance_norm = use_instance_norm

        self.layers = [
            nn.Conv2d(self.in_channel,self.out_channel, kernel_size=(4, 4), stride=2),
            nn.InstanceNorm2d(self.out_channel),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        if not(self.use_instance_norm):
            del self.layers[1]

    def forward(self,inpt):
        tmp = inpt
        for layer in self.layers:
            tmp = layer(tmp)
        outpt = tmp
        return outpt

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = [
            MainBlock(4,64,use_instance_norm=False),
            MainBlock(64,128),
            MainBlock(128,256),
            MainBlock(256,512),
            # nn.Conv2d(512,1,kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(2048,1),
            nn.Sigmoid(),
        ]
    def forward(self,inpt):
        tmp = inpt
        for layer in self.layers:
            tmp = layer(tmp)
        outpt = tmp
        return outpt
