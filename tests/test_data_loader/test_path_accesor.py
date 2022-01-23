import pytest
import os

from style_transfer.data_loader.datasets import data_path_accesor
@pytest.fixture
def path_data():
    path_data = data_path_accesor.get_path_data()
    return path_data

@pytest.fixture
def src_img_paths():
    """source images distribution, to be transformed by GAN"""
    src_img_paths = data_path_accesor.get_src_imgs()
    return src_img_paths

@pytest.fixture
def tgt_img_paths():
    """target images distribution, to be targeted by GAN"""
    tgt_img_paths = data_path_accesor.get_tgt_imgs()
    return tgt_img_paths


def test_if_path_data_not_empty(path_data):
    assert len(os.listdir(path_data)) > 0


def test_if_dir_imgs_not_empty(src_img_paths,tgt_img_paths):
    """
    test if directory of source and target images for Gans, are
    not empty
    :param src_img_paths:
    :param tgt_img_paths:
    :return:
    """
    assert len(list(src_img_paths)) > 0
    assert len(list(tgt_img_paths)) > 0
    for the_list in [src_img_paths,tgt_img_paths]:
        for el in the_list:
            assert el.exists()

