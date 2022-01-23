from random import choice,sample
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
#pytorch modules
from torchvision import transforms
import torch


dir_file = Path(__file__).parent
apple_images = dir_file.joinpath("../../Data/emojis_dataset/emojis_dataset/image/Apple")

#TODO: apple_image should be read from data_path_accesor that should be put outside


def get_random_paths(nb=10):
    """from the apple_images directory"""
    try:
        random_paths = sample(list(apple_images.iterdir()),nb)
    except IndexError:
        raise(f"not image in {apple_images}")
    return random_paths

def get_random_pil_imgs():
    """
        from the apple_images directory
        the pil image is of RGBA format
    """
    ##read the data
    random_paths = get_random_paths()
    pil_imgs_rgba = []
    for random_path in random_paths:
        pil_img = Image.open(random_path)
        pil_img_rgba = pil_img.convert('RGBA')
        pil_imgs_rgba.append(pil_img_rgba)
    return pil_imgs_rgba


def get_random_tensor():
    """
        from the apple_images directory
        the returned tensor is of shape
        (4,*size_image)
    """
    pil_imgs = get_random_pil_imgs()
    tsr = [ transforms.ToTensor()(pil_img) for pil_img in pil_imgs]
    tsr = torch.stack(tsr)
    return tsr

def show_image_with_transparency(rgba_tsrs,idx):
    """
    for a particular index over the first dimension of
    rgba_tsrs

    :param rgba_tsrs: tensor of 4D shape of the form (N,4,...)
    :param idx: the slice over the first dim < N
    :return:
    """
    rgba_tsr = rgba_tsrs[idx]
    rgb,transparency = rgba_tsr[:3],rgba_tsr[3]
    fig,axs = plt.subplots(1,2)
    nominal_rgb = torch.moveaxis(rgb,0,2)
    axs[0].imshow(nominal_rgb)
    axs[1].imshow(transparency,cmap='gray')
    plt.show()