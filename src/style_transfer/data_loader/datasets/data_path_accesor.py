"""
    module that gives path to data
"""
from pathlib import Path
import os
dir_file = Path(__file__).parent

def get_path_data():
    path_data = dir_file.joinpath("../../../../Data/emojis_dataset")
    return path_data





def get_zenly_emojis():
    path_data = get_path_data()
    path_zenly_emoji = path_data.joinpath("zenly_emojis/Emojis Steven")
    list_zenly_emojis = path_zenly_emoji.iterdir()
    return list_zenly_emojis

def get_facebook_emojis():
    path_data = get_path_data()
    path_apple_emoji = path_data.joinpath("emojis_dataset/image/Facebook")
    list_apple_emojis = path_apple_emoji.iterdir()
    return list_apple_emojis

def get_apple_emojis():
    path_data = get_path_data()
    path_apple_emoji = path_data.joinpath("emojis_dataset/image/Apple")
    list_apple_emojis = path_apple_emoji.iterdir()
    return list_apple_emojis


#TODO: add factory of source and target accesor on
# subsequent versions of code
get_src_imgs = get_apple_emojis
get_tgt_imgs = get_facebook_emojis