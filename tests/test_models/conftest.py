import random,torch
import pytest


from style_transfer import utils
from style_transfer import models



@pytest.fixture(scope="function")
def random_batch():
    """
        random batch of 4D from the dataset,with shape of
        the form (.,4,...)
        There is 4 channels because it is an RGBA image
    """
    #TODO , make the size of the batch controlable
    tsr_img = utils.get_random_tensor()
    return tsr_img




# @pytest.fixture(scope="function",params=range(100))
@pytest.fixture(scope="function",params=range(100))
def random_4D_tensor():
    """
    with determinist number of channels equal to 4
    :return: 4-tuple of form (n_batch,4,n_row,n_col)
    """
    n_batch = random.randint(1, 20)
    n_row = random.randint(50, 80)
    n_col = random.randint(50, 80)
    random_4D_shape =  (n_batch,4,n_row,n_col)
    rand_array = torch.randn(*random_4D_shape)
    return rand_array



@pytest.fixture(scope="module")
def model():
    """
    load the generator model
    :return:
    """
    model = models.generator_mod.Generator()
    return model


@pytest.fixture(scope="module")
def discriminator():
    """
    load the generator model
    :return:
    """
    model = models.discriminator_mod.Discriminator()
    return model
