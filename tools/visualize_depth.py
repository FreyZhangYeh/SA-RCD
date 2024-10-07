from misc import colorize
import data_utils
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import cv2

def visualize_depth(dep_txt,outdir,dis):
    dep_files = data_utils.read_paths(dep_txt)
    dep_id = 0
    for dep_file in tqdm(dep_files):
        dep = data_utils.load_depth(dep_file)
        if dis == "dany_rela":
            min_dep = 0.0
            max_dep = 1.0
        else:
            min_dep = 0.0
            max_dep = 80.0
        dep_map = colorize(dep,min_dep,max_dep)
        save_dir = os.path.join(outdir,dis)
        os.makedirs(save_dir,exist_ok=True)
        Image.fromarray(dep_map).save(os.path.join(save_dir, dis + f"_{dep_id}_depth.png"))
        dep_id += 1

def visualize_depth_single_token(dep_file,outdir,dis):
    file_name = os.path.basename(dep_file)[:-4]
    dep = data_utils.load_depth(dep_file)
    dep[dep>60.0] = 60.0
    if dis == "dany_rela":
        min_dep = 0.0
        max_dep = 1.0
    else:
        min_dep = 0.0
        max_dep = 60.0
    dep_map = colorize(dep,min_dep,max_dep,cmap='viridis')
    save_dir = os.path.join(outdir,dis)
    os.makedirs(save_dir,exist_ok=True)
    Image.fromarray(dep_map).save(os.path.join(save_dir, file_name +"_depth.png"))

def visualize_conf_map_single_token(conf_file,dep_file,outdir,dis):
    file_name = os.path.basename(dep_file)[:-4]
    conf_map = data_utils.load_depth(conf_file)
    dep = data_utils.load_depth(dep_file)
    mask = (dep > 0.0)
    conf_map = conf_map*mask
    min_dep = 0.0
    max_dep = 1.0
    conf_map = colorize(conf_map,min_dep,max_dep,cmap='coolwarm')
    save_dir = os.path.join(outdir,dis)
    os.makedirs(save_dir,exist_ok=True)
    Image.fromarray(conf_map).save(os.path.join(save_dir, file_name +"_depth.png"))


def project_points_to_image_single_token(dep_file,img_file,outdir,ksize,dis,project):
    file_name = os.path.basename(dep_file)[:-4]
    dep = data_utils.load_depth(dep_file)
    img = data_utils.load_image(img_file)
    if dis == "dany_rela":
        min_dep = 0.0
        max_dep = 1.0
    else:
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
    
    if project:
        img = img.astype(np.uint8)
        
        # 将渲染后的深度图叠加到RGB图像上
        mask = dep_dilated > 0
        img[mask] = cv2.addWeighted(img[mask], 0.0, dep_color[mask], 1.0, 0)

        save_dir = os.path.join(outdir,dis)
        os.makedirs(save_dir,exist_ok=True)
        
        # 保存结果图像
        output_path = os.path.join(save_dir, f"{file_name}_projected.png")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    else:
        save_dir = os.path.join(outdir,dis)
        os.makedirs(save_dir,exist_ok=True)
        
        # 保存结果图像
        if "lidar" in dep_file:
            output_path = os.path.join(save_dir, f"{file_name}_lidar.png")
        elif "radar" in dep_file:
            output_path = os.path.join(save_dir, f"{file_name}_radar.png")
            
        cv2.imwrite(output_path, cv2.cvtColor(dep_color, cv2.COLOR_RGB2BGR))
        


if __name__ == '__main__':

    # dep_txt = "/home/zfy/RCMDNet/testing/nuscenes/nuscenes_test_selected_dilation_radar_add.txt"
    # outdir = "/data/zfy_data/nuscenes/virl/srd"
    # dis = "srd_add"
    # #dis = "ours_transfer_fromsimm11_w_dense_loss_1.0_w_multilidar_loss_2.0_192500"
    # visualize_depth(dep_txt,outdir,dis)

    outdir = '/data/zfy_data/nuscenes/virl_for_paper/pipeline/'
    #dis = 'radar_image_reprojected'
    dis =  'dany'                      #'rela_norm_uniform_ransac_80'
    dep_file = "/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/metric/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    visualize_depth_single_token(dep_file,outdir,dis)

    # dep_file = "/data/zfy_data/nuscenes/nuscenes_derived_test/radar_image/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    # img_file = "/data/zfy_data/nuscenes/nuscenes_origin/samples/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295399762404.jpg"
    # outdir = "/data/zfy_data/nuscenes/virl_for_paper/pipeline/"
    # dis = "ori-radar"
    # ksize = 10
    # project = False
    # project_points_to_image_single_token(dep_file,img_file,outdir,ksize,dis,project)

    # outdir = '/data/zfy_data/nuscenes/virl_for_paper/pipeline/'
    # dis = 'conf_map'  
    # conf_file =  "/data/zfy_data/nuscenes/nuscenes_derived_test/conf_map_box/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    # dep_file = "/data/zfy_data/nuscenes/nuscenes_derived_test/radar_dilation_box/scene_101/CAM_FRONT/n008-2018-09-18-14-18-33-0400__CAM_FRONT__1537295416612404.png"
    # visualize_conf_map_single_token(conf_file,dep_file,outdir,dis)
