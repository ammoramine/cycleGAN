from style_transfer.data_loader import dataloader_mod
from style_transfer.data_loader import datasets
import pytest


# @pytest.mark.parametrize("batch_size",32)
@pytest.fixture()
def data_loader():
    d_loader = dataloader_mod.get(batch_size=16)
    return d_loader

def test_data_loader(data_loader):
    for el in data_loader:
        break