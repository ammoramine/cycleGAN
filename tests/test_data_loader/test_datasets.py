"""
    test data_loader for cycle_GAN
"""
import pytest
from style_transfer.data_loader import datasets


# @pytest.fixture()
# def dataset_no_transform():
#     dset = dataset.CycleGanDataset()
#     return dset
#
# @pytest.fixture()
# def dataset_tensor_transform():
#     dset = dataset.CycleGanDataset(dataset.full_transform)
#     return dset

@pytest.fixture(scope="module",params=datasets.list_transforms)
def dataset(request):
    dataset = datasets.get(request.param)
    return dataset

def test_elements_of_dataset_are_2_tuples(dataset):
    for el in dataset:
        assert isinstance(el,tuple)
        assert len(el)==2

