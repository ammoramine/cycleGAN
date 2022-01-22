"""
testing the model
"""
import pytest
from style_transfer import models


def test_residual_block_shape_change(multiple_random_4D_tensor):
    """
    the shape of the output of the residual block must
    be equal to its input.
    :param random_4D_tensor: fixture of 4D shape
    :return:
    """
    print(f"\n checking for {len(multiple_random_4D_tensor)} number of tensors")
    for random_4D_tensor in multiple_random_4D_tensor:
        residual_block = models.residual_block_mod.ResidualBlock(4)
        output = residual_block(random_4D_tensor)
        assert output.shape == random_4D_tensor.shape
        # print(output.shape)

# @pytest.mark.skip()
def test_if_shape_is_the_same(random_batch_one,model):
    # N = len(model.layers)
    print(f"batch size : {random_batch_one.shape}")
    out = model(random_batch_one)
    print(f"output batch size : {out.shape}")
    assert out.shape == random_batch_one.shape