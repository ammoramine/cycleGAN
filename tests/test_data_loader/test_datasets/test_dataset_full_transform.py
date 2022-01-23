"""
    test function, for dataset with full transform
    the following functions, should be tested only for
    the full dataset
"""


def test_elements_of_dataset_have_4_channel(dataset_full_transform):
    """as emoticones are 4D images"""
    for el in dataset_full_transform:
        assert el[0].shape[0] == 4

