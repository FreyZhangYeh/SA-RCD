import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import data_utils
from tqdm import tqdm

class Sobel_Vil(nn.Module):
    def __init__(self):
        super(Sobel_Vil, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def get_grad_map(self,depth):
          
        with torch.no_grad():
            
            depth_grad = self.edge_conv(depth) 
            depth_grad = depth_grad.contiguous().view(2, depth.size(1), depth.size(2))

            depth_dx = depth_grad[0, :, :].contiguous().view_as(depth[0])
            depth_dy = depth_grad[1, :, :].contiguous().view_as(depth[0])

            depth_edge = torch.sqrt(depth_dx ** 2 + depth_dy ** 2)

            depth_edge_np = depth_edge.numpy()
            depth_edge_np = (depth_edge_np - depth_edge_np.min()) / (depth_edge_np.max() - depth_edge_np.min())
                
                
        return depth_edge_np
    
def get_edge_img(dep_txt,dis):
    
    edge_func = Sobel_Vil()

    dep_files = data_utils.read_paths(dep_txt)
    dep_id = 0

    save_dir = os.path.join("/data/zfy_data/nuscenes/nuscenes_derived_test_debug/edges",dis)
    os.makedirs(save_dir,exist_ok=True)

    for dep_file in tqdm(dep_files):

        dep = data_utils.load_depth(dep_file)
        dep = torch.from_numpy(dep).float().unsqueeze(0)
        
        edge = edge_func.get_grad_map(dep)

        plt.imsave(os.path.join(save_dir,f'edge_map{dep_id}.png'), edge, cmap='gray')

        dep_id += 1

if __name__ == '__main__':
    dep_txt = "/home/zfy/radar-camera-fusion-depth/visl/path_txt/nuscenes/nuscenes_test_mini_structralnet-wgrad1_wssim0_wdepth1_multiscale1_sobel-1952000.txt"
    dis = "structralnet-wgrad1_wssim0_wdepth1_multiscale1_sobel-1952000_sub"
    get_edge_img(dep_txt,dis)
        
    