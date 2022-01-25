from torch import nn


from . import generator_mod,discriminator_mod


class CycleGan(nn.Module):
    def __init__(self):
        super(CycleGan, self).__init__()


        self.generator_tgt = generator_mod.Generator()
        self.generator_src = generator_mod.Generator()

        self.discriminator_tgt = discriminator_mod.Discriminator()
        self.discriminator_src = discriminator_mod.Discriminator()

        #TODO: custom generators construction to be added later

    def forward(self,inpt_src,inpt_tgt):

        fake_src = self.generator_src(inpt_tgt)
        fake_tgt = self.generator_tgt(inpt_src)

        src_2_src = self.generator_src(fake_tgt)
        tgt_2_tgt = self.generator_tgt(fake_src)

        # err_recon_src = src_2_src - inpt_src
        # err_recon_tgt = tgt_2_tgt - inpt_tgt

        discri_real_src = self.discriminator_src(inpt_src)
        discri_fake_src = self.discriminator_src(fake_src)

        discri_real_tgt = self.discriminator_tgt(inpt_tgt)
        discri_fake_tgt = self.discriminator_tgt(fake_tgt)


        return src_2_src,tgt_2_tgt,discri_real_src,discri_fake_src,discri_real_tgt,discri_fake_tgt

        #TODO: make cycleGAN smaller, and add correponding tests
