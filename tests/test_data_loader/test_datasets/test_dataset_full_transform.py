"""
    test function, for dataset with full transform
    the following functions, should be tested only for
    the full dataset
"""


def test_elements_of_dataset_have_4_channel(dataset_full_transform):
    """as emoticones are 4D images"""
    for el in dataset_full_transform:
        assert el[0].shape[0] == 4


#TODO : resolve the problem of dataset with
# different size of images, by proposing architectural
# solution and easy solutions, with shape transformation
def test_elements_have_the_same_shape(dataset_full_transform):
    firstel = dataset_full_transform[0]
    shape_first_rgba_image = firstel[0].shape
    for el in dataset_full_transform:
        assert el[0].shape == shape_first_rgba_image