from . import data_path_accesor,dataset

list_transforms = ["with_no_transform", "with_full_transform"]


def get(type_dataset="with_full_transform",index=0):
    """
    dataset with specific transform
    :param type_dataset:
    :return:
    """
    if type_dataset == "with_no_transform":
        dset = dataset.CycleGanDataset(index=index)
    elif type_dataset == "with_full_transform":
        dset = dataset.CycleGanDataset(dataset.full_transform,index=index)
    else:
        raise ValueError(f"type_dataset must be among of the {len(list_transforms)} possible values :  \n"+"".join([f"{el}\n" for el in list_transforms]))
    return dset