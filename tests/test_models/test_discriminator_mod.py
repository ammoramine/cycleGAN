"""
    testing the discriminator
"""

import torch
import pytest

@pytest.fixture(scope="function")
def discriminator_input_output(discriminator,random_batch):
    outpt = discriminator(random_batch)
    return (random_batch,outpt)

def test_discriminator_output_shape(discriminator_input_output):
    """
    the shape of the output tensror should be two dimensional
    of the form (N,1), where N is the size of the batch
    :param discriminator_input_output:
    """
    inpt,outpt = discriminator_input_output
    assert outpt.shape[1] == 1
    assert outpt.shape[0] == inpt.shape[0]


def test_discriminator_output_value(discriminator_input_output):
    """
    it must be slower than 1
    :return:
    """
    inpt,outpt = discriminator_input_output
    assert torch.min(outpt) > 0
    assert torch.max(outpt) < 1.0
