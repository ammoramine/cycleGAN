from style_transfer.losses import losses
import pytest
import torch
import random

# @pytest.fixture
def get_random_batch_discrimination_output():
    """
    get a tensor whose shape is of the same form,
    as the output of one of the discriminator of a GAN
    , i.e (N,1) where N int >1
    :return:
    """
    N = random.randint(10,40) #size of batch
    rand_batch = torch.rand(N,1)
    rand_batch = torch.clip(rand_batch,0.0,1.0)
    return rand_batch

@pytest.fixture(scope="function")
def random_input_output():
    random_input = get_random_batch_discrimination_output()
    random_output = get_random_batch_discrimination_output()
    return random_input,random_output

def test_loss_value(random_input_output):
    random_input,random_output = random_input_output
    loss_func = losses.CycleGanLoss()
    loss_val = loss_func(random_input,random_output)
    assert type(loss_val) is torch.Tensor
    assert len(loss_val.shape) == 0 #the output of loss is scalar
    assert loss_val>0 # the loss value must be greater than 0