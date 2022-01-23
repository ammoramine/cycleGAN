"""
    testing the discriminator
"""

import torch

def test_if_discriminator_output_shape_and_value(discriminator,random_batch):
    """
    the shape of the output tensror should be two dimensional
    of the form (N,1), where N is the size of the shape
    the value must be between 0 and 1
    :param discriminator:
    """
    print(f"batch of shape {random_batch.shape}")
    outpt = discriminator(random_batch)
    print(f"shape of output {outpt.shape}")
    assert outpt.shape[1] == 1
    assert outpt.shape[0] == random_batch.shape[0]
    # print(outpt)
    assert torch.min(outpt) > 0
    assert torch.max(outpt) < 1.0
