import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from utils import data_utils, eval_utils
from datasets import datasets_fusionnet as datasets
from utils.log_utils import log
from fusionnet_model import FusionNetModel
from fusionnet_transforms import Transforms
from utils.net_utils import OutlierRemoval
from tqdm import tqdm
from datetime import datetime
import random
from utils.misc import colorize
from utils.data_utils import hist
from utils.eval_utils import apply_thr
import os
import glob
import cv2

def test_fusionnet(output_txt,gt_txt,min_evaluate_depth,max_evaluate_depth):
    output_files = data_utils.read_paths(output_txt)
    gt_files = data_utils.read_paths(gt_txt)
    id = 0
    n_sample = len(output_files)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    for output_file in tqdm(output_files):
        output = data_utils.load_depth(output_file)
        gt = data_utils.load_depth(gt_files[id])

        validity_map = np.where(gt > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            gt > min_evaluate_depth,
            gt < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        # Compute validation metrics
        mae[id] = eval_utils.mean_abs_err(1000.0 * output[mask], 1000.0 * gt[mask])
        rmse[id] = eval_utils.root_mean_sq_err(1000.0 * output[mask], 1000.0 * gt[mask])
        imae[id] = eval_utils.inv_mean_abs_err(0.001 * output[mask], 0.001 * gt[mask])
        irmse[id] = eval_utils.inv_root_mean_sq_err(0.001 * output[mask], 0.001 * gt[mask])

        id = id + 1

    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    print('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'))
    
    print('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        max_evaluate_depth,
        mae,
        rmse,
        imae,
        irmse))

def test_rcdpt(output_txt,gt_txt,min_evaluate_depth,max_evaluate_depth):
    output_files = data_utils.read_paths(output_txt)
    gt_files = data_utils.read_paths(gt_txt)
    id = 0
    n_sample = len(output_files)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    for output_file in tqdm(output_files):
        output = data_utils.load_depth(output_file)
        output = cv2.resize(output, (1600, 768), interpolation=cv2.INTER_LINEAR)
        gt = data_utils.load_depth(gt_files[id])[132:,:]

        validity_map = np.where(gt > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            gt > min_evaluate_depth,
            gt < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        # Compute validation metrics
        mae[id] = eval_utils.mean_abs_err(1000.0 * output[mask], 1000.0 * gt[mask])
        rmse[id] = eval_utils.root_mean_sq_err(1000.0 * output[mask], 1000.0 * gt[mask])
        imae[id] = eval_utils.inv_mean_abs_err(0.001 * output[mask], 0.001 * gt[mask])
        irmse[id] = eval_utils.inv_root_mean_sq_err(0.001 * output[mask], 0.001 * gt[mask])

        id = id + 1

    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    print('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'))
    
    print('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        max_evaluate_depth,
        mae,
        rmse,
        imae,
        irmse))

def get_text(directory):
    png_files = glob.glob(os.path.join(directory, "*.png"))

    with open("/home/zfy/RCMDNet/testing/rcdpt_results.txt", "w") as f:
        for file in png_files:
            f.write(file + "\n")


if __name__ == '__main__':
    # output_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_adabins_predicted_metric.txt"
    # gt_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_lidar.txt"
    # min_evaluate_depth = 0.0
    # max_evaluate_depth = 80.0

    # test_fusionnet(output_txt,gt_txt,min_evaluate_depth,max_evaluate_depth)

    output_txt = "/home/zfy/RCMDNet/testing/rcdpt_results.txt"
    gt_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_lidar.txt"
    min_evaluate_depth = 0.0
    max_evaluate_depth = 80.0

    test_rcdpt(output_txt,gt_txt,min_evaluate_depth,max_evaluate_depth)


