from torch import nn
import torch


class CycleGanLoss():
    def __init__(self):
        self.bce_loss = nn.BCELoss()
    def __call__(self,inpt,target):
        """
        compute  the GAN loss,
        :param inpt: batch of data produced by one of the
        generators, and passed through the correponding discriminator

        :param target: batch of data of targeted disibution by the
        corresponding generator for the 'inpt' data, passed throught
        the corresponding discriminator

        Remark : inpt and target can have shapes of the form (.,1),
        and can be of different size
        :return:
        """
        total_loss = self.bce_loss(1-inpt,torch.ones_like(inpt))
        total_loss += self.bce_loss(target,torch.ones_like(target))
        return total_loss





