'''
Author: Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf
@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import numpy as np

# def apply_thr(quasi_depth,conf_map,thr):
#     #valid_mask = torch.sigmoid(1e3 * (conf_map - thr)).float()
#     valid_mask = (conf_map>thr).float()
#     quasi_depth = quasi_depth*valid_mask
#     while(quasi_depth[quasi_depth>1e-3].shape == 0):
#         thr=thr-0.05
#         #valid_mask = torch.sigmoid(1e3 * (conf_map - thr)).float()
#         valid_mask = (conf_map>thr).float()
#         quasi_depth = quasi_depth*valid_mask
#     return quasi_depth

def apply_thr_2thr(quasi_depth, conf_map, long_dis_from, thr1=0.5, thr2=0.7):
    mask1 = quasi_depth <= long_dis_from
    mask2 = quasi_depth > long_dis_from

    valid_mask1 = (conf_map > thr1).float() * mask1.float()
    valid_mask2 = (conf_map > thr2).float() * mask2.float()

    valid_mask = valid_mask1 + valid_mask2

    quasi_depth = quasi_depth * valid_mask
    
    return quasi_depth

def apply_thr(quasi_depth, conf_map, thr):

    valid_mask = (conf_map > thr).float() 

    quasi_depth = quasi_depth * valid_mask
    
    return quasi_depth

def apply_thr_np_2thr(quasi_depth, conf_map, long_dis_from, thr1=0.5, thr2=0.7):
    mask1 = quasi_depth <= long_dis_from
    mask2 = quasi_depth > long_dis_from

    valid_mask1 = np.logical_and(mask1,conf_map > thr1)
    valid_mask2 = np.logical_and(mask2,conf_map > thr2)

    valid_mask = valid_mask1 + valid_mask2

    quasi_depth = quasi_depth * valid_mask
    
    return quasi_depth

def apply_thr_np_1thr(quasi_depth, conf_map, thr1=0.5):

    valid_mask = conf_map > thr1

    quasi_depth = quasi_depth * valid_mask
    
    return quasi_depth

def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error
    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error
    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error
    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error
    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs((1.0 / tgt) - (1.0 / src)))

def mean_abs_rel_err(src, tgt):
    '''
    Mean absolute relative error (normalize absolute error)
    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute relative error between source and target
    '''

    return np.mean(np.abs(src - tgt) / tgt)