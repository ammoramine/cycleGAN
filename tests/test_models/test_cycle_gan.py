import pytest
import numpy as np

from style_transfer.models import cycle_gan_mod
from style_transfer.utils import utils

@pytest.fixture()
def cycle_gan():
    model = cycle_gan_mod.CycleGan()
    return model

def test_if_shape_is_the_same(cycle_gan):
    inpt_src,inpt_tgt = utils.get_random_tensor(),utils.get_random_tensor()
    print(f"batch shape for input src's distribution : {inpt_src.shape}")
    print(f"batch shape for input target's distribution : {inpt_tgt.shape}")
    out = cycle_gan(inpt_src,inpt_tgt)
    assert out[0].shape == inpt_src.shape
    assert out[1].shape == inpt_tgt.shape

    #TODO: make there two assertions simpler
    assert len(np.unique([el.shape[0] for el in out])) == 1
    assert out[2].shape == out[3].shape == out[4].shape == out[5].shape
    # batch_size must be the same all over