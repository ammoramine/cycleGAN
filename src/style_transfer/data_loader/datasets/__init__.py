from . import data_path_accesor,dataset



def get(type_dataset):
    """
    dataset with specific transform
    :param type_dataset:
    :return:
    """
    list_options = ["with_no_transform","with_full_transform"]
    if type_dataset == "with_no_transform":
        dset = dataset.CycleGanDataset()
    elif type_dataset == "with_full_transform":
        dset = dataset.CycleGanDataset(dataset.full_transform)
    else:
        raise ValueError(f"type_dataset must be among values "+" ".join([f"{el}" for el in list_options]))