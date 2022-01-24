from . import residual_block_mod
from torch import nn

class Generator(nn.Module):
    def __init__(self,nb_residual_block=3):
        super(Generator, self).__init__()
        self.nb_residual_block = nb_residual_block

        layer1 = nn.Conv2d(4,32,kernel_size=(9,9))
        layer2 = nn.Conv2d(32,64,kernel_size=(3,3))
        layer3 = nn.Conv2d(64,128,kernel_size=(3,3))
        residual_blocks = [residual_block_mod.ResidualBlock(128) for _ in range(self.nb_residual_block)]
        layerm3 = nn.ConvTranspose2d(128,64,kernel_size=(3,3))
        layerm2 = nn.ConvTranspose2d(64,32,kernel_size=(3,3))
        layerm1 = nn.ConvTranspose2d(32,4,kernel_size=(9,9))
        self.layers = nn.ModuleList([layer1,layer2,layer3]+residual_blocks+[layerm3,layerm2,layerm1])

    def forward(self,inpt):
        tmp = inpt
        for layer in self.layers:
            tmp = layer(tmp)
        out = tmp
        return out