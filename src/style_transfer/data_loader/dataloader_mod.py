from torch.utils.data import DataLoader
from . import datasets

def get(batch_size,index):
    """
        construct datalaoder with specific batch_size
        for src ou target image (index=0 or 1), with shuffled data

    :param batch_size: int
    :param index: 0 or 1, if equal 0 dataloader or src image, otherwise of target iamge
    :return:
    """
    dataset = datasets.get(type_dataset="with_full_transform",index=index)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return dataloader