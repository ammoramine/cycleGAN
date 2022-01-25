import pytest

from style_transfer.models import cycle_gan_mod
from style_transfer.trainer_and_val import common
from style_transfer.losses import losses
from style_transfer.data_loader import dataloader_mod
def test_data_loader_param_initialization():
    params = common.init_params_for_trainning()

    assert type(params.model) is cycle_gan_mod.CycleGan

    assert type(params.loss) is losses.CycleGanLoss

    assert type(params.data_loaders.src_dataloader) is dataloader_mod.DataLoader
    assert type(params.data_loaders.tgt_dataloader) is dataloader_mod.DataLoader
