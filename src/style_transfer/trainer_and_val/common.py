import torch

from .. import models
from ..data_loader import dataloader_mod
from ..losses import losses
from ..utils import dict_to_obj
#TODO: add a singleton for device
device = "cuda" if torch.cuda.is_available() else "cpu"

#TODO: add conversion to state dictionnaries

def init_params_for_trainning():
    params = dict()
    params["model"] = models.cycle_gan_mod.CycleGan()
    params["data_loaders"] = init_data_loaders()
    params["loss"] = losses.CycleGanLoss()
    return dict_to_obj.dict2obj(params)

#TODO: take into consideration, the fact that src and target
# distribution don't have necessarily the same shape

def init_data_loaders(batch_size=32):
    params = dict()
    params["src_dataloader"] = dataloader_mod.get(batch_size,index=0)
    params["tgt_dataloader"] = dataloader_mod.get(batch_size,index=1)

    return params
