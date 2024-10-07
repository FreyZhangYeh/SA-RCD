import numpy as np
import os, time
import multiprocessing as mp
import estimator
import data_utils
from tqdm import tqdm


def process_scale(sparse_depth_file, mono_pred_file,
                  max_pred = 80.0, min_pred = 0.0,
                  max_depth = 80.0, min_depth = 1.0,
                  save_path = None,
                  depth_type = 'pos',
                  mode = 'ls',
                  idx = 0):

    print(f"start processing {idx}")

    sparse_depth = data_utils.load_depth(sparse_depth_file).astype(np.float32)
    mono_pred = data_utils.load_depth(mono_pred_file).astype(np.float32)

    sparse_depth_valid = (sparse_depth < max_depth) * (sparse_depth > min_depth)
    sparse_depth_valid = sparse_depth_valid.astype(bool)

    if depth_type == 'inv':
        sparse_depth[~sparse_depth_valid] = np.inf
        sparse_depth = 1.0 / sparse_depth
    else:
        sparse_depth[~sparse_depth_valid] = 0.0


    # global scale and shift alignment
    GlobalAlignment = estimator.LeastSquaresEstimator(
        estimate=mono_pred,
        target=sparse_depth,
        valid=sparse_depth_valid
    )
    if mode == 'uniform_ransac':
        # uniform ransac
        GlobalAlignment.compute_scale_and_shift_ransac_uniform(num_iterations=400, sample_size=5,
                                                inlier_threshold=6, inlier_ratio_threshold=0.9)
    elif mode == 'ransac':
        # ransac
        GlobalAlignment.compute_scale_and_shift_ran(num_iterations=400, sample_size=5,
                                                inlier_threshold=6, inlier_ratio_threshold=0.9)
    elif mode == 'ls':
        # least square
        GlobalAlignment.compute_scale_and_shift()
        
    GlobalAlignment.apply_scale_and_shift()

    # # only global scale alignment
    # GlobalAlignment = estimator.Optimizer(
    #     estimate=mono_pred,
    #     target=sparse_depth,
    #     valid=sparse_depth_valid,
    #     depth_type=depth_type
    # )
    # timestart = time.time()
    # GlobalAlignment.optimize_scale()
    # time_use = time.time() - timestart
    # GlobalAlignment.apply_scale()

    GlobalAlignment.clamp_min_max(clamp_min=min_pred, clamp_max=max_pred)
    int_depth = GlobalAlignment.output.astype(np.float32)
    if depth_type == 'inv':
        int_depth = 1.0 / int_depth

    scene = sparse_depth_file.split("/")[6]
    camera = sparse_depth_file.split("/")[7]
    filename = os.path.basename(sparse_depth_file)
    dany_alignmented_dir = os.path.join(save_path,scene,camera)
    os.makedirs(dany_alignmented_dir, exist_ok=True)
    dany_alignmented_path = os.path.join(dany_alignmented_dir,filename[:filename.rfind('.')] + '.png')

    if save_path:
        data_utils.save_depth(int_depth, os.path.join(save_path, dany_alignmented_path))

    print(f"finish processing {idx}")

    return dany_alignmented_path



if __name__ == '__main__':
    np.random.seed(355123027)
    mode = 'uniform_ransac'                  #"uniform_ransac"
    start_id = 0
    
    sparse_depth_file = '/data/zfy_data/nuscenes/nuscenes_derived_test/radar_image/scene_1/CAM_FRONT/n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533153900862404.png'
    mono_pred_file = '/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/relative_origin/scene_1/CAM_FRONT/n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533153900862404.png'
    save_path = '/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/rela_radar_alignmented_uniform_ransac_400-5'
    
    process_scale(
                sparse_depth_file=sparse_depth_file, 
                mono_pred_file=mono_pred_file, 
                save_path=save_path,
                mode=mode
            )

   