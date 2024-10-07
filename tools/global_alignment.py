import numpy as np
import os, time
import multiprocessing as mp
import estimator
import data_utils
from tqdm import tqdm


def process_scale(args):

    sparse_depth_file,mono_pred_file,max_pred,min_pred,max_depth,min_depth,save_path,mono_input,mode,idx,norm_dany = args

# def process_scale(
#         sparse_depth_file, mono_pred_file,
#                   max_pred = 120.0, min_pred = 0.0,
#                   max_depth = 120.0, min_depth = 1.0,
#                   save_path = None,
#                   mode = 'ls',
#                   idx = 0,
#                   mono_input = 'relative_origin',
#                   norm_dany = None,
#                   save_depth = False
#                   ):

    print(f"start processing {idx}")

    sparse_depth = data_utils.load_depth(sparse_depth_file).astype(np.float32)
    mono_pred_rela = data_utils.load_depth(mono_pred_file).astype(np.float32)

    sparse_depth_valid = (sparse_depth < max_depth) * (sparse_depth > min_depth)
    sparse_depth_valid = sparse_depth_valid.astype(bool)

    # if depth_type == 'inv':
    #     sparse_depth[~sparse_depth_valid] = np.inf
    #     sparse_depth = 1.0 / sparse_depth
    # else:
    #     sparse_depth[~sparse_depth_valid] = 0.0

    if mono_input == 'heyoeyo':
        
        if norm_dany:
            mono_pred = (mono_pred_rela - mono_pred_rela.min())/(mono_pred_rela.max() - mono_pred_rela.min())
            sparse_depth[~sparse_depth_valid] = np.inf
            sparse_depth = 1.0 / sparse_depth
            inlier_threshold = 0.015
            
        else:
            mono_pred_rela[mono_pred_rela <= (1.0 / max_depth)] = 1.0 / max_depth
            mono_pred_rela_valid = (mono_pred_rela > 0).astype(bool)
            mono_pred_rela[~mono_pred_rela_valid] = np.inf
            mono_pred = 1.0 / mono_pred_rela
            inlier_threshold = 6.0


    if mono_input == 'LiheYoung':

        if norm_dany:
            mono_pred_rela = (mono_pred_rela - mono_pred_rela.min())/(mono_pred_rela.max() - mono_pred_rela.min())
            mono_pred_rela_valid = (mono_pred_rela > 0).astype(bool)
            mono_pred_rela[~mono_pred_rela_valid] = np.inf
            mono_pred = 1.0 / mono_pred_rela

        inlier_threshold = 6.0

    # global scale and shift alignment
    GlobalAlignment = estimator.LeastSquaresEstimator(
        estimate=mono_pred,
        target=sparse_depth,
        valid=sparse_depth_valid
    )
    if mode == 'uniform_ransac':
        # uniform ransac
        GlobalAlignment.compute_scale_and_shift_ransac_uniform(num_iterations=400, sample_size=5,
                                                inlier_threshold=inlier_threshold, inlier_ratio_threshold=0.9)
    elif mode == 'ransac':
        # ransac
        GlobalAlignment.compute_scale_and_shift_ran(num_iterations=400, sample_size=5,
                                                inlier_threshold=inlier_threshold, inlier_ratio_threshold=0.9)
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
    # if depth_type == 'inv':
    #     int_depth = 1.0 / int_depth
    if mono_input == 'heyoeyo':
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

def parallel_process(sparse_depth_files, mono_pred_files, save_path, mode, start_id=0, num_processes=4,mono_input='relative_original',norm_dany = True,max_pred = 255.0,min_pred = 0.0,max_depth = 80.0,min_depth = 1.0):

    dany_alignmented_paths = []

    if num_processes == 1:
        # 单进程顺序处理
        for i in tqdm(range(start_id,len(sparse_depth_files))):
            sparse_depth_file = sparse_depth_files[i]
            mono_pred_file = mono_pred_files[i]
            dany_alignmented_path = process_scale(
                sparse_depth_file=sparse_depth_file, 
                mono_pred_file=mono_pred_file, 
                save_path=save_path,
                mode=mode,
                mono_input=mono_input,
                norm_dany=norm_dany,
                max_pred = max_pred,min_pred = min_pred,max_depth = max_depth,min_depth = min_depth
            )
            dany_alignmented_paths.append(dany_alignmented_path)
    else:
        # 多进程并行处理
        pool_results = []
        pool_inputs = []
        pool = mp.Pool(num_processes)
        for i in range(start_id,len(sparse_depth_files)):
            sparse_depth_file = sparse_depth_files[i]
            mono_pred_file = mono_pred_files[i]

            inputs = [
                sparse_depth_file,
                mono_pred_file,
                max_pred,
                min_pred,
                max_depth,
                min_depth,
                save_path,
                mono_input,
                mode,
                i,
                norm_dany
            ]

            pool_inputs.append(inputs)
        
        pool_results = pool.map(process_scale, pool_inputs)
        for result in pool_results:
            dany_alignmented_paths.append(result)

        pool.close()

    return dany_alignmented_paths


if __name__ == '__main__':
    np.random.seed(355123027)
    mode = "ls"
    train_test = "train"
    mono_input = "heyoeyo"
    norm_dany = True

    max_pred = 120.0;min_pred = 0.0
    max_depth = 120.0;min_depth = 1.0
    
    if train_test == 'train':
        #train
        sparse_depth_root = '/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub/nuscenes_train_radar_image.txt'
        mono_pred_root ='/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub/nuscenes_train_dany_rela_predicted.txt'
        save_path = '/data/zfy_data/nuscenes/nuscenes_derived/dany_predicted_sub/' + mono_input + '_radar_alignmented_' + mode +'_400-5'
        txt_save_path = os.path.join('/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub','nuscenes_train_dany_' + mono_input + '_radar_aligned_' + mode + "_400-5.txt")
        
    elif train_test == 'test':
        #test
        sparse_depth_root = '/home/zfy/radar-camera-fusion-depth/testing/nuscenes/nuscenes_test_radar_image.txt'
        mono_pred_root ='/home/zfy/radar-camera-fusion-depth/testing/nuscenes/nuscenes_test_dany_rela_predicted.txt'
        save_path = '/data/zfy_data/nuscenes/nuscenes_derived_test/dany_predicted/' + mono_input + '_radar_alignmented_' + mode +'_400-5'
        txt_save_path = os.path.join('/home/zfy/radar-camera-fusion-depth/testing/nuscenes','nuscenes_test_dany_' + mono_input + '_radar_aligned_' + mode + "_400-5.txt")
        
    sparse_depth_files = data_utils.read_paths(sparse_depth_root)
    mono_pred_files = data_utils.read_paths(mono_pred_root)
    
    path_only = False
    num_processes = 10
    start_id = 0
   
    assert len(sparse_depth_files) == len(mono_pred_files)
    
    print(f"sparse_depth_root:{sparse_depth_root}")
    print(f"mono_pred_root:{mono_pred_root}")
    print(f"save_path:{save_path}")
    print(f"txt_save_path:{txt_save_path}")
    print(f"mode:{mode}")
    print(f"start_id:{start_id}")
    print(f"mono_input:{mono_input}")
    print(f"norm_dany:{norm_dany}")
    print(f"max_pred:{max_pred}")
    print(f"min_pred:{min_pred}")
    print(f"max_depth:{max_depth}")
    print(f"min_depth:{min_depth}")
    
    
    
    dany_alignmented_paths = parallel_process(sparse_depth_files, mono_pred_files, save_path, mode, start_id, num_processes,mono_input,norm_dany,max_pred,min_pred,max_depth,min_depth)

    if save_path:
        data_utils.write_paths(txt_save_path,dany_alignmented_paths)

    # num_processes = 4 # mp.cpu_count()-4

    # pool = mp.Pool(num_processes)
    # sum = 0.0
    # cnt = 0

    # dany_alignmented_paths = []

    # if num_processes == 1:
    
    #     for i in tqdm(range(0,len(sparse_depth_files))):
    #         sparse_depth_file,mono_pred_file = sparse_depth_files[i],mono_pred_files[i]
    #         dany_alignmented_path = process_scale(sparse_depth_file=sparse_depth_file, mono_pred_file=mono_pred_file, save_path=save_path, path_only=path_only)
    #         dany_alignmented_paths.append(dany_alignmented_path)
    
    # else:

    #     for i in tqdm(range(0,len(sparse_depth_files))):
    #         sparse_depth_file,mono_pred_file = sparse_depth_files[i],mono_pred_files[i]
    #         result = pool.apply_async(process_scale, args=(sparse_depth_file, mono_pred_file, save_path, path_only))
    #         dany_alignmented_paths.append(result.get())


    # pool.close()
    # pool.join()

    # data_utils.write_paths(os.path.join("/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub","nuscenes_train_dany_metric_radar_aligned_ransac_400-5.txt"),dany_alignmented_paths)

    # # average_scale
    # average_scale = sum / cnt
    # print('sum: ', sum)
    # print('average: ', average_scale)

    # dpt 0.8117274480821013
    # midas 0.0013031250057613435
