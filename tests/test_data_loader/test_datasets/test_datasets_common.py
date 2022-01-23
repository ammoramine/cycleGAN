"""
    test functions for dataset with all possible values of
    src_transform
"""



def test_elements_of_dataset_are_2_tuples(dataset):
    for el in dataset:
        assert isinstance(el,tuple)
        assert len(el)==2


