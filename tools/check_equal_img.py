from misc import colorize
import data_utils
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import cv2

def check_equal_img(src_path,tgt_path):
    src = data_utils.load_depth(src_path)
    tgt = data_utils.load_depth(tgt_path)
    if np.array_equal(src, tgt):
        print("src = tgt")
    else:
        print("src != tgt")
    

if __name__ == '__main__':
    src_path = "/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/radar_alignmented_ransac_test_400_prelist/scene_0/CAM_FRONT/n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533153857912404.png"
    tgt_path = "/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/radar_alignmented_ransac_test_400/scene_0/CAM_FRONT/n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533153857912404.png"
    check_equal_img(src_path,tgt_path)

