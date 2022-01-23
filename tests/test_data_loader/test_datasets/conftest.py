import pytest
from style_transfer.data_loader import datasets


@pytest.fixture(scope="module",params=datasets.list_transforms)
def dataset(request):
    dataset = datasets.get(request.param)
    return dataset


@pytest.fixture(scope="module")
def dataset_full_transform():
    dataset = datasets.get("with_full_transform")
    return dataset