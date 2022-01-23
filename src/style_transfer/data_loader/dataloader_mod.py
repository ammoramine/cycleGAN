from torch.utils.data import DataLoader
from . import datasets

def get(batch_size,index):
    """dataloder"""
    dataset = datasets.get(type_dataset="with_full_transform",index=index)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return dataloader