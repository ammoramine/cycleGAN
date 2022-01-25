from style_transfer.data_loader import dataloader_mod



def get_cycle_gan_input(batch_size=32):
    d_loader_src = dataloader_mod.get(batch_size, 0)
    d_loader_tgt = dataloader_mod.get(batch_size, 1)
    for (inpt_src,inpt_tgt) in zip(d_loader_src,d_loader_tgt):
        break
    return inpt_src[0],inpt_tgt[0]