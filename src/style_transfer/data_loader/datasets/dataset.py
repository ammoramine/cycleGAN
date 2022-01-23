from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from itertools import chain

from style_transfer.data_loader.datasets import data_path_accesor

dir_file = Path(__file__).parent
path_data = dir_file.joinpath("../../../Data")

full_transform = transforms.Compose([Image.open, transforms.ToTensor()])


class CycleGanDataset(Dataset):
    def __init__(self,src_transform=None):
        super(CycleGanDataset, self).__init__()
        self.src_imgs_path = list(data_path_accesor.get_src_imgs())
        self.tgt_imgs_path = list(data_path_accesor.get_tgt_imgs())
        self.src_transform = src_transform

        self.all_imgs_with_lbl = self.get_all_imgs_with_lbl()

    def get_all_imgs_with_lbl(self):
        """
            the target image's distribution is labelled with 1
            and the src image's distribution is labelled with 0
        """
        imgs = self.src_imgs_path+self.tgt_imgs_path
        lbls = len(self.src_imgs_path)*[0]+len(self.tgt_imgs_path)*[1]
        return list(zip(imgs,lbls))

    def __getitem__(self, i):
        inpt,label = self.all_imgs_with_lbl[i]
        if self.src_transform is not None:
            inpt = self.src_transform(inpt)
        return inpt,label
