from style_transfer.data_loader import dataloader_mod
from style_transfer.data_loader import datasets
import pytest
import torch


@pytest.fixture(params=[0,1])
def data_loader(request):
    index = request.param
    d_loader = dataloader_mod.get(batch_size=16,index=index)
    return d_loader

@pytest.mark.slow
def test_data_loader_is_shuffled(data_loader):
    firstel = [el for el in data_loader][0]
    firstel_of_new_loading = [el for el in data_loader][0]
    assert torch.any(firstel_of_new_loading[0] != firstel[0]) or torch.any(firstel_of_new_loading[1] != firstel[1])

def test_dataloader_data_of_float_type(data_loader):
    for el in data_loader:
        assert el[0].dtype == torch.float32
        assert el[1].dtype == torch.float32