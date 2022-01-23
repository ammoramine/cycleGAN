from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from itertools import chain
import torch

from style_transfer.data_loader.datasets import data_path_accesor

dir_file = Path(__file__).parent
path_data = dir_file.joinpath("../../../Data")


#TODO: store this function in a proper module

def convert_to_rgba(pil_img):
    return pil_img.convert('RGBA')

full_transform = transforms.Compose([Image.open, convert_to_rgba, transforms.ToTensor()])






class CycleGanDataset(Dataset):
    def __init__(self,src_transform=None,index=0):
        """

        :param src_transform: transform to be applied
        to the list of images
        :param index: equals 0 or 1 ,
        if equal 0, the dataset is dataset of src images
        otherwise it is dataset of target images
        """
        super(CycleGanDataset, self).__init__()
        assert index in [0,1]
        self.index = index
        if self.index==0:
            self.path_imgs = list(data_path_accesor.get_src_imgs())
        elif self.index==1:
            self.path_imgs = list(data_path_accesor.get_tgt_imgs())

        self.src_transform = src_transform
        self.all_imgs_with_lbl = self.get_all_imgs_with_lbl()

    def get_all_imgs_with_lbl(self):
        """
            the target image's distribution is labelled with 1
            and the src image's distribution is labelled with 0
        """
        imgs = self.path_imgs
        lbls = len(self.path_imgs)*[float(self.index)]
        return list(zip(imgs,lbls))

    def __getitem__(self, i):
        inpt,label = self.all_imgs_with_lbl[i]
        if self.src_transform is not None:
            inpt = self.src_transform(inpt)
        return inpt,torch.tensor(label)

    def __len__(self):
        return len(self.all_imgs_with_lbl)