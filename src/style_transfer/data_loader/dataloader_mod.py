from torch.utils.data import DataLoader
from . import datasets

def get(batch_size):
    """dataloder"""
    dataset = datasets.get(type_dataset="with_full_transform")
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return dataloader