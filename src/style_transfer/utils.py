from random import choice
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
#pytorch modules
from torchvision import transforms
import torch


dir_file = Path(__file__).parent
apple_images = dir_file.joinpath("../../Data/emojis_dataset/image/Apple")

def get_random_path():
    """from the apple_images directory"""
    try:
        random_path = choice(list(apple_images.iterdir()))
    except IndexError:
        raise(f"not image in {apple_images}")
    return random_path

def get_random_pil_img():
    """
        from the apple_images directory
        the pil image is of RGBA format
    """
    ##read the data
    random_path = get_random_path()
    pil_img = Image.open(random_path)
    pil_img_rgba = pil_img.convert('RGBA')
    return pil_img_rgba


def get_random_tensor():
    """
        from the apple_images directory
        the returned tensor is of shape
        (4,*size_image)
    """
    pil_img = get_random_pil_img()
    tsr = transforms.ToTensor()(pil_img)
    return tsr

def show_image_with_transparency(rgba_tsr):
    rgb,transparency = rgba_tsr[:3],rgba_tsr[3]
    fig,axs = plt.subplots(1,2)
    nominal_rgb = torch.moveaxis(rgb,0,2)
    axs[0].imshow(nominal_rgb)
    axs[1].imshow(transparency,cmap='gray')
    plt.show()