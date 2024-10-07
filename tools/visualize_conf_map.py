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
from matplotlib.colors import Normalize,LinearSegmentedColormap


def visualize_results(image, confidence_map, radar_points, filename='result_visualization.png'):
    plt.figure(figsize=(16, 9))
    
    if image.dtype == np.float32:
        image = image / 255.0

    norm = Normalize(vmin=0.0,vmax=1.0)
    # confidence_map = np.log1p(confidence_map * 50) / np.log1p(50)
    # colors = [(0, 0, 0.5), (1, 1, 0.5), (1, 0, 0)]  # 深蓝 -> 浅黄 -> 红
    # n_bins = 100  # Discretizes the interpolation into bins
    # cmap_name = 'custom_cmap'
    # cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
    plt.imshow(image, alpha=1)
    plt.scatter(radar_points[:, 0], radar_points[:, 1], c='yellow', s=1.0, marker='o')
    plt.imshow(confidence_map, cmap='coolwarm', alpha=0.4, interpolation='none',norm=norm)
    plt.colorbar(label='Confidence Level', orientation='vertical')
    plt.axis('off')  # 去掉坐标轴
    plt.title('Conf_Map on Camera Image')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def process_single_frame(args):
    idx, image_path, conf_map_path, radar_points_path, radar_dilation_path, save_depth_dir = args
    image = data_utils.load_image(image_path)
    conf_map = data_utils.load_depth(conf_map_path)
    # conf_map[conf_map<=0.7] = 0.1
    radar_points = np.load(radar_points_path)

    radar_dilation = data_utils.load_depth(radar_dilation_path)
    conf_map_mask = (radar_dilation>0.0)
    conf_map = conf_map*conf_map_mask

    scene = conf_map_path.split("/")[6]
    camera = conf_map_path.split("/")[7]
    filename = os.path.basename(conf_map_path)
    conf_map_virl_dir = os.path.join(save_depth_dir, scene, camera)
    os.makedirs(conf_map_virl_dir, exist_ok=True)
    conf_map_virl_path = os.path.join(conf_map_virl_dir, filename[:filename.rfind('.')] + '.png')
    visualize_results(image, conf_map, radar_points, conf_map_virl_path)

    return conf_map_virl_path

def process(image_path, conf_map_path, radar_points_path, radar_dilation_path, save_depth_dir,max_workers=4):
    image_paths = data_utils.read_paths(image_path)
    conf_map_paths = data_utils.read_paths(conf_map_path)
    radar_points_paths = data_utils.read_paths(radar_points_path)
    radar_dilation_paths = data_utils.read_paths(radar_dilation_path)
    assert len(image_paths) == len(conf_map_paths) == len(radar_points_paths) == len(radar_dilation_paths)
    conf_map_virl_paths = []
    tasks = [(idx, image_paths[idx], conf_map_paths[idx], radar_points_paths[idx], radar_dilation_paths[idx], save_depth_dir) for idx in range(len(image_paths))]
    with Pool(max_workers) as pool:
        for conf_map_virl_path in tqdm(pool.imap(process_single_frame, tasks), total=len(image_paths)):
            conf_map_virl_paths.append(conf_map_virl_path)

if __name__ == '__main__':
    image_path = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_image.txt"
    radar_points_path = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_radar.txt"
    conf_map_path = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_conf.txt"
    radar_dilation_path = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_selected_dilation_radar.txt"
    save_depth_dir = "/data/zfy_data/nuscenes/virl/conf_map_custom_dilation_mask"
    max_workers = 10 # 这里设置最大并行进程数
    process(image_path=image_path, 
            radar_points_path=radar_points_path,
            conf_map_path=conf_map_path,
            radar_dilation_path=radar_dilation_path,
            save_depth_dir=save_depth_dir, 
            max_workers=max_workers)



