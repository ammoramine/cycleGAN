import random,torch
import pytest


from style_transfer import utils
from style_transfer import models



@pytest.fixture(scope="function")
def random_batch_one():
    """
        random batch with one element from the dataset
    """
    tsr_img = utils.get_random_tensor().unsqueeze(0)
    return tsr_img

def get_random_4D_tensor():
    """with a fixed number of 4 channels"""
    n_batch = random.randint(1, 20)
    n_row = random.randint(50, 80)
    n_col = random.randint(50, 80)
    rand_array = torch.randn(n_batch, 4, n_row, n_col)
    return rand_array

@pytest.fixture(scope="function")
def random_4D_tensor():
    rand_array = get_random_4D_tensor()
    return rand_array

@pytest.fixture(scope="function")
# @pytest.mark.parametrize('nb', 100)
def multiple_random_4D_tensor(random_batch_one):
    """
    :return:
    """
    nb = 100
    random_batches = [get_random_4D_tensor() for _ in range(nb)]
    return random_batches


@pytest.fixture(scope="module")
def model():
    """
    load the generator model
    :return:
    """
    model = models.generator_mod.Generator()
    return model

