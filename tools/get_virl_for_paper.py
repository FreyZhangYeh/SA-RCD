from misc import colorize
import data_utils
import os
from PIL import Image
from tqdm import tqdm
from utils.eval_utils import apply_thr_np_1thr,apply_thr_np_2thr
from misc import colorize
import data_utils
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import cv2
from multiprocessing import Pool, Manager
import matplotlib.colors as mcolors
from scipy.ndimage import label
import shutil

def get_srd(rdep_txt,conf_txt,outdir):
    rdep_files = data_utils.read_paths(rdep_txt)
    conf_files = data_utils.read_paths(conf_txt)
    dep_id = 0
    min_dep = 0.0
    max_dep = 80.0
    
    for rdep_file in tqdm(rdep_files):
        rdep = data_utils.load_depth(rdep_file)
        conf = data_utils.load_depth(conf_files[dep_id])
        srd = apply_thr_np_2thr(quasi_depth=rdep, conf_map=conf, long_dis_from=50.0, thr1=0.5, thr2=0.4)

        dep_map = colorize(srd,min_dep,max_dep)
        
        filename = os.path.splitext(os.path.basename(rdep_file))[0] + '.png'     
        scene = rdep_file.split("/")[6]
        camera = rdep_file.split("/")[7]
        output_depth_radar_dirpath = os.path.join(outdir,scene,camera)
        
        os.makedirs(output_depth_radar_dirpath,exist_ok=True)
        Image.fromarray(dep_map).save(os.path.join(output_depth_radar_dirpath, filename))
        dep_id += 1

def get_rd(rdep_txt,outdir):
    rdep_files = data_utils.read_paths(rdep_txt)
    dep_id = 0
    min_dep = 0.0
    max_dep = 80.0
    
    for rdep_file in tqdm(rdep_files):
        rdep = data_utils.load_depth(rdep_file)

        dep_map = colorize(rdep,min_dep,max_dep)
        
        filename = os.path.splitext(os.path.basename(rdep_file))[0] + '.png'     
        scene = rdep_file.split("/")[6]
        camera = rdep_file.split("/")[7]
        output_depth_radar_dirpath = os.path.join(outdir,scene,camera)
        
        os.makedirs(output_depth_radar_dirpath,exist_ok=True)
        Image.fromarray(dep_map).save(os.path.join(output_depth_radar_dirpath, filename))
        dep_id += 1

def get_lidar(ldep_txt,outdir,ksize):
    rdep_files = data_utils.read_paths(ldep_txt)
    dep_id = 0
    min_dep = 0.0
    max_dep = 80.0
    
    for rdep_file in tqdm(rdep_files):
        rdep = data_utils.load_depth(rdep_file)

                # 膨胀深度图以突出非零点
        kernel = np.ones((ksize, ksize), np.uint8)
        dep_dilated = cv2.dilate(rdep, kernel, iterations=1)

        dep_map = colorize(dep_dilated,min_dep,max_dep,cmap='viridis')
        
        # # 使用jet色图将深度值映射到颜色
        # norm = Normalize(vmin=min_dep, vmax=max_dep)
        # colormap = plt.get_cmap('viridis')
        # dep_map = colormap(norm(dep_dilated))
        
        filename = os.path.splitext(os.path.basename(rdep_file))[0] + '.png'     
        scene = rdep_file.split("/")[6]
        camera = rdep_file.split("/")[7]
        output_depth_radar_dirpath = os.path.join(outdir,scene,camera)
        
        os.makedirs(output_depth_radar_dirpath,exist_ok=True)
        Image.fromarray(dep_map).save(os.path.join(output_depth_radar_dirpath, filename))
        dep_id += 1

def project_depth_to_image_single_token(args):
    img_file,dep_file,savedir,ksize = args
    file_name = os.path.splitext(os.path.basename(dep_file))[0] + '.png'     
    scene = dep_file.split("/")[6]
    camera = dep_file.split("/")[7]
    outdir = os.path.join(savedir,scene,camera)

    dep = data_utils.load_depth(dep_file)
    img = data_utils.load_image(img_file)

    min_dep = 0.0
    max_dep = 80.0

    # 膨胀深度图以突出非零点
    kernel = np.ones((ksize, ksize), np.uint8)
    dep_dilated = cv2.dilate(dep, kernel, iterations=1)
    
    # 使用jet色图将深度值映射到颜色
    norm = Normalize(vmin=min_dep, vmax=max_dep)
    colormap = plt.get_cmap('viridis')
    dep_color = colormap(norm(dep_dilated))
    
    # 转换为0-255的RGB格式
    dep_color = (dep_color[:, :, :3] * 255).astype(np.uint8)
    
    img = img.astype(np.uint8)
    
    # 将渲染后的深度图叠加到RGB图像上
    mask = dep_dilated > 0
    img[mask] = cv2.addWeighted(img[mask], 0.0, dep_color[mask], 1.0, 0)

    os.makedirs(outdir,exist_ok=True)
    
    # 保存结果图像
    output_path = os.path.join(outdir, file_name)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def process_project_depth_to_image(savedir, image_path_txt, depth_path_txt, ksize, max_workers=4):

    image_paths = data_utils.read_paths(image_path_txt)
    depth_paths = data_utils.read_paths(depth_path_txt)

    assert len(image_paths) == len(depth_paths)


    tasks = [(image_paths[idx], depth_paths[idx], savedir, ksize) for idx in range(len(depth_paths))]
    with Pool(max_workers) as pool:
        list(tqdm(pool.imap_unordered(project_depth_to_image_single_token, tasks), total=len(depth_paths)))

def project_region_to_image_single_token(args):
    img_file,dep_file,radar_file,ksize,savedir = args
    file_name = os.path.splitext(os.path.basename(dep_file))[0] + '.png'     
    scene = dep_file.split("/")[6]
    camera = dep_file.split("/")[7]
    outdir = os.path.join(savedir,scene,camera)

    dep = data_utils.load_depth(dep_file)
    img = data_utils.load_image(img_file)
    radar = data_utils.load_depth(radar_file)
    region = np.where(dep > 0, 1.0, 0.0)

    min_region = 0.0
    max_region = 1.0

    min_radar = 0.0
    max_radar = 80.0
    
    kernel = np.ones((ksize, ksize), np.uint8)
    radar_d = cv2.dilate(radar, kernel, iterations=1)

    # 使用jet色图将深度值映射到颜色
    norm_d = Normalize(vmin=min_region, vmax=max_region)
    # colormap_d = plt.get_cmap('twilight')
    colormap_d = mcolors.ListedColormap(['none', (237/255.0, 141/255.0, 90/255.0)]) 
    region_color = colormap_d(norm_d(region))

    norm_r = Normalize(vmin=min_radar, vmax=max_radar)
    colormap_r = plt.get_cmap('viridis')
    radar_color = colormap_r(norm_r(radar_d))


    
    # 转换为0-255的RGB格式
    region_color = (region_color[:, :, :3] * 255).astype(np.uint8)
    radar_color = (radar_color[:, :, :3] * 255).astype(np.uint8)
    
    img = img.astype(np.uint8)
    
    # 将渲染后的深度图叠加到RGB图像上
    mask = region > 0
    img[mask] = cv2.addWeighted(img[mask], 0.5, region_color[mask], 0.5, 0)
    
    mask_r = radar_d > 0
    img[mask_r] = cv2.addWeighted(img[mask_r], 0.0, radar_color[mask_r], 1.0, 0)

    os.makedirs(outdir,exist_ok=True)
    
    # 保存结果图像
    output_path = os.path.join(outdir, file_name)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def filter_small_regions(image, min_size):
    """
    过滤掉二值图像中小于指定大小的连通区域。

    Args:
        image (numpy.ndarray): 输入的二值图像（0表示背景，1表示前景）。
        min_size (int): 保留的最小区域大小（以像素为单位）。

    Returns:
        numpy.ndarray: 过滤后的图像，仅保留大于等于min_size的区域，其他区域设为0。
    """
    # 标记连通区域
    labeled_array, num_features = label(image)
    
    # 计算每个区域的大小
    region_sizes = np.bincount(labeled_array.ravel())
    
    # 使用布尔掩码保留大于等于min_size的区域
    mask = np.isin(labeled_array, np.where(region_sizes >= min_size)[0])
    
    # 生成过滤后的图像
    filtered_image = np.where(mask, image, 0)
    
    return filtered_image

def process_project_region_to_image(savedir, image_path_txt, depth_path_txt, radar_path_txt, ksize, max_workers=4):

    image_paths = data_utils.read_paths(image_path_txt)
    depth_paths = data_utils.read_paths(depth_path_txt)
    radar_paths = data_utils.read_paths(radar_path_txt)

    assert len(image_paths) == len(depth_paths) == len(radar_paths)


    tasks = [(image_paths[idx], depth_paths[idx], radar_paths[idx], ksize, savedir) for idx in range(len(depth_paths))]
    with Pool(max_workers) as pool:
        list(tqdm(pool.imap_unordered(project_region_to_image_single_token, tasks), total=len(depth_paths)))

def project_conf_to_image_single_token(args):
    img_file,dep_file,radar_file,conf_file,ksize,savedir = args
    file_name = os.path.splitext(os.path.basename(dep_file))[0] + '.png'     
    scene = conf_file.split("/")[6]
    camera = conf_file.split("/")[7]
    outdir = os.path.join(savedir,scene,camera)

    conf = data_utils.load_depth(conf_file)
    img = data_utils.load_image(img_file)
    radar = data_utils.load_depth(radar_file)
    radar_dilation = data_utils.load_depth(dep_file)
    region =  np.where((radar_dilation > 0), 1.0, 0.0)
    region = filter_small_regions(region, 200)
    conf = np.where((region > 0), conf, 0.0)
    
    # conf = np.where((conf >= 0.5), conf, 0.0)

    min_conf = 0.0
    max_conf = 1.0

    min_radar = 0.0
    max_radar = 80.0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    radar_d = cv2.dilate(radar, kernel, iterations=1)

    mask_radar_d = radar_d > 0
    
    conf = colorize(value=conf,vmin=min_conf,vmax=max_conf,cmap='coolwarm')
    radar_d = colorize(value=radar_d,vmin=min_radar,vmax=max_radar,cmap='viridis')

    conf_color = (conf[:, :, :3]).astype(np.uint8)
    radar_color = (radar_d[:, :, :3]).astype(np.uint8)
    
    img = img.astype(np.uint8)

    img = cv2.addWeighted(img, 0.5, conf_color, 0.5, 0)
    img[mask_radar_d] = cv2.addWeighted(img[mask_radar_d], 0.0, radar_color[mask_radar_d], 1.0, 0)

    os.makedirs(outdir,exist_ok=True)
    
    # 保存结果图像
    output_path = os.path.join(outdir, file_name)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def process_project_conf_to_image(savedir, image_path_txt, depth_path_txt, radar_path_txt,conf_path_txt,ksize, max_workers=4):

    image_paths = data_utils.read_paths(image_path_txt)
    depth_paths = data_utils.read_paths(depth_path_txt)
    radar_paths = data_utils.read_paths(radar_path_txt)
    conf_paths = data_utils.read_paths(conf_path_txt)

    assert len(image_paths) == len(depth_paths) == len(radar_paths) == len(conf_paths)


    tasks = [(image_paths[idx], depth_paths[idx], radar_paths[idx], conf_paths[idx], ksize, savedir) for idx in range(len(depth_paths))]
    with Pool(max_workers) as pool:
        list(tqdm(pool.imap_unordered(project_conf_to_image_single_token, tasks), total=len(depth_paths)))

def process_image_files(savedir,image_path_txt, depth_path_txt):
    image_paths = data_utils.read_paths(image_path_txt)
    depth_paths = data_utils.read_paths(depth_path_txt)

    assert len(image_paths) == len(depth_paths)

    idx = 0


    for image_file in tqdm(image_paths):
        # image = data_utils.load_image(image_file)
        dep_file = depth_paths[idx]
        file_name = os.path.splitext(os.path.basename(dep_file))[0] + '.jpg'     
        scene = dep_file.split("/")[6]
        camera = dep_file.split("/")[7]

        image_save_dir = os.path.join(savedir,scene,camera)
        os.makedirs(image_save_dir,exist_ok=True)
        
        image_save_path = os.path.join(image_save_dir,file_name)
        shutil.copy(image_file,image_save_path)

        idx+=1
        
def visualize_depth(dep_txt,outdir):
    dep_files = data_utils.read_paths(dep_txt)
    dep_id = 0
    for dep_file in tqdm(dep_files):
        dep = data_utils.load_depth(dep_file)
        file_name = os.path.splitext(os.path.basename(dep_file))[0] + '.png'     
        scene = dep_file.split("/")[6]
        camera = dep_file.split("/")[7]

        min_dep = 0.0
        max_dep = 80.0
        
        dep_map = colorize(dep,min_dep,max_dep)
        save_dir = os.path.join(outdir,scene,camera)
        os.makedirs(save_dir,exist_ok=True)
        Image.fromarray(dep_map).save(os.path.join(save_dir, file_name))
        dep_id += 1

if __name__ == '__main__':

    # outdir = "/data/zfy_data/nuscenes/virl_for_paper/mono_depth"
    # dep_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_dany_metric_predicted.txt"
    # visualize_depth(dep_txt,outdir)
    
    outdir = "/data/zfy_data/nuscenes/virl_for_paper/r"
    dep_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_radar_image.txt"
    get_lidar(dep_txt,outdir,10)

    # savedir = "/data/zfy_data/nuscenes/nuscenes_derived_test/image"
    # image_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_image.txt"
    # depth_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_selected_dilation_radar.txt"
    # process_image_files(savedir,image_path_txt,depth_path_txt)

    # conf_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_conf_mono-add.txt"
    # outdir = "/data/zfy_data/nuscenes/virl/srd_mono/add-preserve-long-dis-0.4"
    # get_srd(rdep_txt,conf_txt,outdir)

    # outdir = "/data/zfy_data/nuscenes/virl_for_paper/rd"
    # get_rd(rdep_txt,outdir)

    #dep_file = "/data/zfy_data/nuscenes/nuscenes_derived_test/radar_image/scene_20/CAM_FRONT/n008-2018-08-01-15-34-25-0400__CAM_FRONT__1533152655912404.png"
    # dep_file = "/data/zfy_data/nuscenes/nuscenes_derived_test/radar_dilation_box/scene_100/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295399762404.png"
    # img_file = "/data/zfy_data/nuscenes/nuscenes_origin/samples/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295399762404.jpg"
    # outdir = "/data/zfy_data/nuscenes/virl/paper/architecture"
    # dis = "radar_dilation"
    # ksize = 1
    # project = True
    # project_points_to_image_single_token(dep_file,img_file,outdir,ksize,dis,project)
    
    # savedir = "/data/zfy_data/nuscenes/virl_for_paper/quasidepth_radarnet"
    # image_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_image.txt"
    # depth_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_output_depth.txt"
    # ksize = 1
    # max_workers = 12
    # process_project_depth_to_image(savedir, image_path_txt, depth_path_txt, ksize, max_workers)

    # savedir = "/data/zfy_data/nuscenes/virl_for_paper/dregion_on_image_pink"
    # image_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_image.txt"
    # depth_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_radar_dilation_box.txt"
    # radar_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_radar_image.txt"
    # max_workers = 12
    # ksize = 10
    # process_project_region_to_image(savedir, image_path_txt, depth_path_txt, radar_path_txt, ksize, max_workers)
    
    # ldep_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_lidar.txt"
    # outdir = "/data/zfy_data/nuscenes/virl_for_paper/lidar_gt"
    # kszie = 10
    # get_lidar(ldep_txt,outdir,kszie)
    
    # savedir = "/data/zfy_data/nuscenes/virl_for_paper/conf_on_image/ours-filtertiny"
    # image_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_image.txt"
    # depth_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_selected_dilation_radar_box.txt"
    # radar_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_radar_image.txt"
    # conf_path_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_conf_attwp.txt"
    # max_workers = 12
    # ksize = 8
    # process_project_conf_to_image(savedir, image_path_txt, depth_path_txt, radar_path_txt, conf_path_txt, ksize, max_workers)
