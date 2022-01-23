from style_transfer import models

def test_residual_block_shape_doesnt_change(random_4D_tensor):
    """
    the shape of the output of the residual block must
    be equal to its input.
    :param random_4D_tensor: fixture of 4D shape
    :return:
    """
    print(f"\n checking for tensor of shape {random_4D_tensor.shape} ")
    residual_block = models.residual_block_mod.ResidualBlock(4)
    output = residual_block(random_4D_tensor)
    # print(output.shape)
    assert output.shape == random_4D_tensor.shape