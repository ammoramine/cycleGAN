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
    params["models"] = init_models()
    params["data_loaders"] = init_data_loaders()
    params["losses"] = losses.CycleGanLoss()
    return dict_to_obj.dict2obj(params)

#TODO: take into consideration, the fact that src and target
# distribution don't have necessarily the same shape

def init_models():
    params = dict()
    params["generator_direct"] = models.generator_mod.Generator()
    # generator from the source to the target distribution
    params["generator_reverse"] = models.generator_mod.Generator()
    # generator from the target to the source distribution
    params["discriminator_src"] = models.discriminator_mod.Discriminator()
    # discriminator for  the source distribution
    params["discriminator_tgt"] = models.discriminator_mod.Discriminator()
    # discriminator for  the target distribution
    return params

def init_data_loaders(batch_size=32):
    params = dict()
    params["src_dataloader"] = dataloader_mod.get(batch_size,index=0)
    params["tgt_dataloader"] = dataloader_mod.get(batch_size,index=1)

    return params
