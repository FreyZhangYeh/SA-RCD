import sys
sys.path.append("/home/zfy/RCMDNet/src")
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood
from utils import data_utils
from tqdm import tqdm
import os
import matplotlib.colors as mcolors
from multiprocessing import Pool, Manager
from matplotlib.colors import Normalize
import cv2

def region_growing_skimage(mono_depth, seed_point, threshold=0.5):
    seed_y, seed_x = seed_point
    region_mask = flood(mono_depth, (seed_y, seed_x), connectivity=2, tolerance=threshold)
    return region_mask

def crop_region_mask(region_mask, center, patch_size):
    h, w = patch_size
    cy, cx = center
    start_y = max(0, cy - h // 2)
    end_y = min(region_mask.shape[0], cy + h // 2)
    start_x = max(0, cx - w // 2)
    end_x = min(region_mask.shape[1], cx + w // 2)
    
    cropped_region_mask = np.zeros_like(region_mask)
    cropped_region_mask[start_y:end_y, start_x:end_x] = region_mask[start_y:end_y, start_x:end_x]
    
    return cropped_region_mask

def generate_confidence_map_with_region_growing_single_frame(radar_image, mono_depth, gt, depth_threshold=0.4, region_threshold=0.5):
    assert radar_image.shape == gt.shape == mono_depth.shape, "Shape of radar_image, mono_depth, and gt must be the same"
    confidence_map = np.zeros_like(gt)
    dilation_radar_depth = np.zeros_like(gt)
    radar_points = np.argwhere(radar_image > 0)
    region_masks = np.zeros_like(mono_depth)

    for id, point in enumerate(radar_points):
        y, x = point
        radar_depth = radar_image[y, x]
        region_mask = region_growing_skimage(mono_depth, (y, x), region_threshold)
        region_masks += region_mask

        dilation_radar_depth[region_mask] = radar_depth

        local_gt = gt * region_mask
        diff = np.abs(local_gt - radar_depth)
        patch_confidence_map = np.where(diff <= depth_threshold, 1, 0)
        confidence_map += patch_confidence_map * region_mask

    confidence_map = np.where(gt > 0, confidence_map, 0)
    region_masks = np.where(region_masks > 0.0, 1.0, 0.0)
    return confidence_map, region_masks, radar_points, dilation_radar_depth

def virl_dilation(radar_image, mono_depth, gt, depth_threshold=0.4, region_threshold=0.5,patch_size=(200, 200)):
    assert radar_image.shape == gt.shape == mono_depth.shape, "Shape of radar_image, mono_depth, and gt must be the same"
    confidence_map = np.zeros_like(gt)
    dilation_radar_depth = np.zeros_like(gt)
    radar_points = np.argwhere(radar_image > 0)
    region_masks = np.zeros_like(mono_depth)
    visited = np.zeros_like(gt, dtype=bool)

    min_region = 0.0
    max_region = 1.0

    min_radar = 0.0
    max_radar = 80.0

    for id, point in enumerate(radar_points):
        y, x = point
        radar_depth = radar_image[y, x]
        region_mask = region_growing_skimage(mono_depth, (y, x), region_threshold)

        # Crop the region mask to the specified patch size
        cropped_region_mask = crop_region_mask(region_mask, (y, x), patch_size)
        region_masks += cropped_region_mask

        norm_d = Normalize(vmin=min_region, vmax=max_region)
        colormap_d = mcolors.ListedColormap(['none', (237/255.0, 141/255.0, 90/255.0)]) 
        region_color = colormap_d(norm_d(region_masks))

        # Use the cropped region mask to update visited locations
        visited_region = cropped_region_mask & ~visited
        visited |= cropped_region_mask

        dilation_radar_depth[visited_region] = radar_depth

        norm_r = Normalize(vmin=min_radar, vmax=max_radar)
        colormap_r = plt.get_cmap('viridis')
        radar_color = colormap_r(norm_r(dilation_radar_depth))

        region_color = (region_color[:, :, :3] * 255).astype(np.uint8)
        radar_color = (radar_color[:, :, :3] * 255).astype(np.uint8)

        local_gt = gt * visited_region
        diff = np.abs(local_gt - radar_depth)
        patch_confidence_map = np.where(diff <= depth_threshold, 1, 0)

        confidence_map += patch_confidence_map * visited_region

        cv2.imwrite(os.path.join("/data/zfy_data/nuscenes/virl_for_paper/dilation_process/roi",f"{id}.png"), cv2.cvtColor(region_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join("/data/zfy_data/nuscenes/virl_for_paper/dilation_process/dr",f"{id}.png"), cv2.cvtColor(radar_color, cv2.COLOR_RGB2BGR))

    confidence_map = np.where(gt > 0, confidence_map, 0)
    region_masks = np.where(region_masks > 0.0, 1.0, 0.0)

    # Normalize the confidence map to be within [0, 1]
    confidence_map = np.clip(confidence_map, 0, 1)

    unique_values = np.unique(confidence_map)
    assert np.all((unique_values == 0) | (unique_values == 1)), f"conf_label 包含非 0 和 1 的值: {unique_values}"
    
    return confidence_map, region_masks, radar_points,dilation_radar_depth

def visualize_results(image, mono_depth, confidence_map, region_masks, radar_points, filename='result_visualization.png'):
    plt.figure(figsize=(18, 4))
    region_cmap = mcolors.ListedColormap(['none', 'orange'])
    conf_gt_cmap = mcolors.ListedColormap(['none', 'red'])
    if image.dtype == np.float32:
        image = image / 255.0
    plt.subplot(1, 2, 1)
    plt.imshow(mono_depth, cmap='jet', alpha=1)
    plt.scatter(radar_points[:, 1], radar_points[:, 0], c='yellow', s=1.0, marker='o')
    plt.imshow(region_masks, cmap=region_cmap, alpha=0.4, interpolation='none')
    plt.imshow(confidence_map, cmap=conf_gt_cmap, alpha=0.5, interpolation='none')
    plt.title('Mono Depth with Radar Points, Mask Regions, and Confidence Maps Label')
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=1)
    plt.scatter(radar_points[:, 1], radar_points[:, 0], c='yellow', s=1.0, marker='o')
    plt.imshow(region_masks, cmap=region_cmap, alpha=0.4, interpolation='none')
    plt.imshow(confidence_map, cmap=conf_gt_cmap, alpha=0.5, interpolation='none')
    plt.title('Image with Radar Points, Mask Regions, and Confidence Maps Label')
    plt.savefig(filename)
    plt.close()

def process_single_frame(args):
    idx, image_path, radar_path, mono_path, gt_path, save_depth_dir, depth_threshold, region_threshold, save_virl, tag = args
    image = data_utils.load_image(image_path)
    radar = data_utils.load_depth(radar_path)
    mono = data_utils.load_depth(mono_path)
    gt = data_utils.load_depth(gt_path)
    conf_gt, region_masks, radar_points, dilation_radar_depth= virl_dilation(
        radar, mono, gt, depth_threshold, region_threshold)
    scene = radar_path.split("/")[6]
    camera = radar_path.split("/")[7]
    filename = os.path.basename(radar_path)
    conf_gt_dir = os.path.join(save_depth_dir, 'conf_gt', scene, camera)
    dilation_radar_depth_dir = os.path.join(save_depth_dir, 'radar_dilation', scene, camera)
    region_masks_dir = os.path.join(save_depth_dir, 'region_masks', scene, camera)
    os.makedirs(conf_gt_dir, exist_ok=True)
    os.makedirs(dilation_radar_depth_dir, exist_ok=True)
    os.makedirs(region_masks_dir, exist_ok=True)
    conf_gt_path = os.path.join(conf_gt_dir, filename[:filename.rfind('.')] + '.png')
    dilation_radar_depth_path = os.path.join(dilation_radar_depth_dir, filename[:filename.rfind('.')] + '.png')
    region_masks_path = os.path.join(region_masks_dir,filename[:filename.rfind('.')] + '.png')
    data_utils.save_depth(conf_gt, conf_gt_path)
    data_utils.save_depth(dilation_radar_depth, dilation_radar_depth_path)
    data_utils.save_depth(region_masks,region_masks_path)
    if save_virl:
        virl_dir = os.path.join(save_depth_dir.replace("nuscenes_derived","nuscenes_derived_debug") + '_virl', scene, camera) if tag == "train" else \
                   os.path.join(save_depth_dir.replace("nuscenes_derived_test","nuscenes_derived_test_debug") + '_virl', scene, camera)
        os.makedirs(virl_dir, exist_ok=True)
        virl_gt_path = os.path.join(virl_dir, filename[:filename.rfind('.')] + '.png')
        visualize_results(image, mono, conf_gt, region_masks, radar_points, virl_gt_path)
    return conf_gt_path,dilation_radar_depth_path

def process(tag,save_virl, image_path, radar_image_path, mono_depth_path, gt_path, save_depth_dir, depth_threshold=0.4, region_threshold=0.5, max_workers=4):
    image_paths = data_utils.read_paths(image_path)
    radar_paths = data_utils.read_paths(radar_image_path)
    mono_paths = data_utils.read_paths(mono_depth_path)
    gt_paths = data_utils.read_paths(gt_path)
    assert len(radar_paths) == len(mono_paths) == len(gt_paths) == len(image_paths)
    conf_gt_paths = []
    dilation_radar_depth_paths = []
    tasks = [(idx, image_paths[idx], radar_paths[idx], mono_paths[idx], gt_paths[idx], save_depth_dir, depth_threshold, region_threshold, save_virl, tag) for idx in range(len(radar_paths))]
    with Pool(max_workers) as pool:
        for conf_gt_path,dilation_radar_depth_path in tqdm(pool.imap(process_single_frame, tasks), total=len(radar_paths)):
            conf_gt_paths.append(conf_gt_path)
            dilation_radar_depth_paths.append(dilation_radar_depth_path)
    data_utils.write_paths("/home/zfy/RCMDNet/" + tag + "ing" + "/nuscenes/nuscenes_" + tag + "_conf_gt_ls.txt", conf_gt_paths)
    data_utils.write_paths("/home/zfy/RCMDNet/" + tag + "ing" + "/nuscenes/nuscenes_" + tag + "_radar_dilation_ls.txt", dilation_radar_depth_paths)    

if __name__ == '__main__':

    idx = 0
    tag = "train"
    image_path = "/data/zfy_data/nuscenes/nuscenes_origin/samples/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.jpg"
    radar_path = "/data/zfy_data/nuscenes/nuscenes_derived_test/radar_image/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    mono_path = "/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/metric/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    gt_path = "/data/zfy_data/nuscenes/nuscenes_derived_test/ground_truth_interp/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    save_depth_dir = "/data/zfy_data/nuscenes/nuscenes_derived" if tag == "train" else "/data/zfy_data/nuscenes/nuscenes_derived_test"
    dis = "radar_dilation_adabins"
    save_depth_dir = os.path.join(save_depth_dir,dis)
    depth_threshold = 0.4
    region_threshold = 0.2
    save_virl = False

    args = idx, image_path, radar_path, mono_path, gt_path, save_depth_dir, depth_threshold, region_threshold, save_virl, tag 
    process_single_frame(args)


def visualize_outlier(image,region_mask,radar_point,idx):
    if image.dtype == np.float32:
        image = image / 255.0

    plt.figure(figsize=(20, 20))
    region_cmap = mcolors.ListedColormap(['none', 'grey'])
    
    plt.imshow(image,alpha=1)
    plt.scatter(radar_point[1], radar_point[0], c='yellow', s=1.0, marker='o')
    plt.imshow(region_mask, cmap=region_cmap, alpha=0.8, interpolation='none')
    plt.title('Image with Radar Point and its coresp Region')
    
    filename = os.path.join("/home/zfy/RCMDNet/visl/region_overlay/",str(idx) + "_.png")
    plt.savefig(filename)

