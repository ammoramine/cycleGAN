from torch import nn
import torch


class CycleGanLoss():
    def __init__(self,lbda=0.1,verbose=False):
        """

        :param
        lbda: parameters to control the importance of
        the reconstrucion/cycle loss
        :param verbose: if set to True,show the total
        loss when the __call__ is applied, besides the
        discrimination loss and the cyclic loss
        """
        self.bce_loss = nn.BCELoss()
        self.lbda = lbda
        self.verbose = verbose

    def __call__(self,pred_model,target_model):
        """
        compute the loss for the training of he cyclegan

        :param pred_model:
        a 6-tuple, containing in the order
        src_2_src : of 4D dimension, with 4 channel, application on the source the generator from src to target then target to src
        tgt_2_tgt : of 4D dimension, with 4 channel, ,application on the target the generator from taret to arc then  src to target
        discri_real_src : of 2D dim , of the form (N,1) application on batch of source images, the discriminator (on source) network
        discri_real_tgt : of 2D dim , of the form (N,1) application on batch of target images, the discriminator (on target) network
        discri_fake_src : of 2D dim , of the form (N,1) application on batch of generated source images , the discriminator (on source) network
        discri_fake_tgt : of 2D dim , of the form (N,1) application on batch of generated target images , the discriminator (on target) network
        :param target_model:
         a 2-tuple, containing in sequence, a batch of the
        input distribution, and a batch of the output distribution

        Remark : pred and target can have shapes of the form (.,1),
        and can be of different size
        :return:
        """
        src_2_src,tgt_2_tgt,discri_real_src,discri_fake_src,discri_real_tgt,discri_fake_tgt = pred_model

        src,tgt = target_model

        total_discri_loss = self.bce_loss(1-discri_fake_src,torch.ones_like(discri_fake_src))
        total_discri_loss += self.bce_loss(discri_real_src,torch.ones_like(discri_real_src))

        total_discri_loss += self.bce_loss(1-discri_fake_tgt,torch.ones_like(discri_fake_tgt))
        total_discri_loss += self.bce_loss(discri_real_tgt,torch.ones_like(discri_real_tgt))

        total_cycle_loss = torch.sum(torch.abs(src_2_src-src))
        total_cycle_loss += torch.sum(torch.abs(tgt-tgt_2_tgt))

        total_loss = self.lbda*total_cycle_loss + total_discri_loss

        if self.verbose:
            return total_loss,total_discri_loss,total_cycle_loss
        return total_loss

    def verbose(self):
        """set verbose to True"""
        self.verbose = True
    def laconic(self):
        """set verbose to False"""
        self.verbose = False