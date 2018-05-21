import glob
import os
from py_utils import load_utils
import pickle


def getImagePath():
    user_root = os.path.expanduser('~')

    image_path = 'datasets/created_dataset/SOS_Merged'
    return os.path.join(user_root, image_path)

def get_test_list():
    user_root = os.path.expanduser('~')
    image_list_path = 'datasets/created_dataset/split/test.txt'
    image_path = 'datasets/created_dataset/SOS_Merged'

    image_list = load_utils.load_string_list(os.path.join(user_root, image_list_path))
    image_path_list = []
    for s_image_name in image_list:
        s_full_image_path = os.path.join(user_root,image_path,s_image_name)
        if os.path.isfile(s_full_image_path):
            image_path_list.append(s_full_image_path)
    return image_path_list


def get_pdefined_anchors():
    user_root = os.path.expanduser('~')
    pdefined_anchor_file = 'Dev/adobe_pytorch/datasets/pdefined_anchor.pkl'
    pdefined_anchors = pickle.load(open(os.path.join(user_root, pdefined_anchor_file), 'r'))
    return pdefined_anchors
