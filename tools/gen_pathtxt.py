import data_utils
from tqdm import tqdm
import os
import glob
import random

def gen_imgtxt_from_lidartxt(input_txt,output_txt):
    input_paths = data_utils.read_paths(input_txt)
    output_paths = []
    for file in tqdm(input_paths):
        filename = os.path.basename(file)
        filename = filename.replace("png","jpg")
        output_path = os.path.join("/data/zfy_data/nuscenes/nuscenes_origin/samples/CAM_FRONT",filename)
        output_paths.append(output_path)
    data_utils.write_paths(output_txt,output_paths)

def gen_txt_from_lidartxt(input_txt,output_txt):
    input_paths = data_utils.read_paths(input_txt)
    output_paths = []
    for file in tqdm(input_paths):
        filename = os.path.basename(file)
        scene = file.split("/")[6]
        target_dir = os.path.join("/data/zfy_data/nuscenes/nuscenes_derived/lwlr_predicted_sub",scene,"CAM_FRONT")
        output_path = os.path.join(target_dir,filename)
        output_paths.append(output_path)
    data_utils.write_paths(output_txt,output_paths)

def gen_validtxt_for_structral(input_dir,output_dir,num):

    image_txt = os.path.join(input_dir,"nuscenes_train_image.txt")
    dany_txt = os.path.join(input_dir,"nuscenes_train_dany_metirc_predicted.txt")

    image_paths = data_utils.read_paths(image_txt)
    dany_paths = data_utils.read_paths(dany_txt)

    assert len(image_paths) == len(dany_paths)

    step = len(image_paths) // num

    image_paths = image_paths[::step]
    dany_paths = dany_paths[::step]

    output_image_txt = os.path.join(output_dir,"nuscenes_train_image.txt")
    output_dany_txt = os.path.join(output_dir,"nuscenes_train_dany_metirc_predicted.txt")
    
    data_utils.write_paths(output_image_txt,image_paths)
    data_utils.write_paths(output_dany_txt,dany_paths)



def gen_subdataset_fromtxt(input_dir,out_dir,num_of_subscenes,num_of_subsamples):

    lidar_txt = glob.glob(os.path.join(input_dir, "*_lidar.txt"))[0]
    image_txt = glob.glob(os.path.join(input_dir, "*_image.txt"))[0]
    gt_txt = glob.glob(os.path.join(input_dir, "*_ground_truth.txt"))[0]
    gt_itp_txt  = glob.glob(os.path.join(input_dir, "*_ground_truth_interp.txt"))[0]
    radar_txt  = glob.glob(os.path.join(input_dir, "*_radar.txt"))[0]
    radar_rp_txt = glob.glob(os.path.join(input_dir, "*_radar_reprojected.txt"))[0]

    lidar_paths = data_utils.read_paths(lidar_txt)
    image_paths = data_utils.read_paths(image_txt)
    gt_paths = data_utils.read_paths(gt_txt)
    gt_itp_paths = data_utils.read_paths(gt_itp_txt)
    radar_paths = data_utils.read_paths(radar_txt)
    radar_rp_paths = data_utils.read_paths(radar_rp_txt)

    scenes_dict = {}
    total_samples = 0

    for lidar in tqdm(lidar_paths):
        scene = lidar.split("/")[6]
        # filename = os.path.basename(lidar)
        # filename = filename[:filename.rfind('.')]
        
        if scene not in scenes_dict:
            scenes_dict[scene] = [lidar]
        else:
            scenes_dict[scene].append(lidar)
        
        total_samples += 1
        print("{},Sample nums:{}".format(scene,len(scenes_dict[scene])))
        
    print("Total Scenes:{} Total Samples:{}".format(len(scenes_dict),total_samples))
    
    valid_files = []
    path_to_index = {path: index for index, path in enumerate(lidar_paths)}
    
    if len(scenes_dict) > num_of_subscenes:
        selected_scenes = random.sample(list(scenes_dict.keys()), num_of_subscenes)
        reduced_scenes_dict = {scene: scenes_dict[scene] for scene in selected_scenes}
    else:
        reduced_scenes_dict = scenes_dict

    
    for scene in tqdm(reduced_scenes_dict.keys()):
        if len(reduced_scenes_dict[scene]) > num_of_subsamples:
            reduced_scenes_dict[scene] = random.sample(reduced_scenes_dict[scene], num_of_subsamples)

        valid_files.extend([path_to_index[x] for x in reduced_scenes_dict[scene] if x in path_to_index])
        print("{},Sample nums:{}".format(scene,len(reduced_scenes_dict[scene])))
    
    print("Total Reduced Scenes:{} Total Reduced Samples:{}".format(len(reduced_scenes_dict),len(valid_files)))

    # valid_lidar_paths = []
    # valid_image_paths = []
    # valid_gt_paths = []
    # valid_gt_itp_paths = []
    # valid_radar_paths = []
    # valid_radar_rp_paths = []

    # for lidar,image in tqdm(enumerate(lidar_paths,image_paths)):
    #     filename = os.path.basename(lidar)
    #     filename = filename[:filename.rfind('.')]
    #     if filename in valid_files:
    #         valid_lidar = lidar
    #         valid_lidar_paths.append(valid_lidar)
    #         valid_image = image
    #         valid_image_paths.append(valid_image)
    #         valid_gt_path = lidar.replace("lidar","ground_truth")
    #         valid_gt_paths.append(valid_gt_path)
    #         valid_gt_itp_path = lidar.replace("lidar","ground_truth_interp")
    #         valid_gt_itp_paths.append(valid_gt_itp_path)
    #         valid_radar_path = lidar.replace("lidar","radar_points")
    #         valid_radar_paths.append(valid_radar_path)
    #         valid_radar_rp_path = lidar.replace("lidar","radar_points_reprojected")
    #         valid_radar_rp_paths.append(valid_radar_rp_path)

    valid_lidar_paths = [lidar_paths[i] for i in valid_files]
    valid_image_paths = [image_paths[i] for i in valid_files]
    valid_gt_paths = [gt_paths[i] for i in valid_files]
    valid_gt_itp_paths = [gt_itp_paths[i] for i in valid_files]
    valid_radar_paths = [radar_paths[i] for i in valid_files]
    valid_radar_rp_paths = [radar_rp_paths[i] for i in valid_files]

    lidar_save = os.path.join(out_dir,os.path.basename(lidar_txt))
    image_save = os.path.join(out_dir,os.path.basename(image_txt))
    gt_save = os.path.join(out_dir,os.path.basename(gt_txt))
    gt_itp_save = os.path.join(out_dir,os.path.basename(gt_itp_txt))
    radar_save = os.path.join(out_dir,os.path.basename(radar_txt))
    radar_rp_save = os.path.join(out_dir,os.path.basename(radar_rp_txt))

    data_utils.write_paths(lidar_save,valid_lidar_paths)
    data_utils.write_paths(image_save,valid_image_paths)
    data_utils.write_paths(gt_save,valid_gt_paths)
    data_utils.write_paths(gt_itp_save,valid_gt_itp_paths)
    data_utils.write_paths(radar_save,valid_radar_paths)
    data_utils.write_paths(radar_rp_save,valid_radar_rp_paths)

    print("Finish{}samples".format(len(valid_lidar_paths)))
 
def count_png_files(directory):

    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                count += 1
    return count


if __name__ == '__main__':
    # input_txt = "/home/zfy/radar-camera-fusion-depth/testing/nuscenes/nuscenes_test_depth_predicted.txt"
    # output_txt = "/home/zfy/radar-camera-fusion-depth/visl/path_text/nuscenes_test_mini_depth_predicted.txt"
    # gen_subpath_fromtxt(input_txt,output_txt,800,810)
    # input_dir = "/home/zfy/radar-camera-fusion-depth/training/nuscenes"
    # out_dir = "/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub"
    # input_dir = "/home/zfy/radar-camera-fusion-depth/validation/nuscenes"
    # out_dir = "/home/zfy/radar-camera-fusion-depth/validation/nuscenes_sub"
    # num_of_subscenes = 10
    # num_of_subsamples = 20
    # finished_scenes =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479]
    # gen_subdataset_fromtxt(input_dir,out_dir,num_of_subscenes,num_of_subsamples,finished_scenes)
    # input_txt = "/home/zfy/radar-camera-fusion-depth/visl/path_txt/nuscenes/nuscenes_test_mini_lidar.txt"
    # output_txt = "/home/zfy/radar-camera-fusion-depth/visl/path_txt/nuscenes/nuscenes_test_mini_image.txt"
    input_txt = "/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub/nuscenes_train_lidar.txt"
    output_txt = "/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub/nuscenes_train_lwlr_predicted.txt"
    #gen_txt_from_lidartxt(input_txt,output_txt)
    # count = count_png_files("/data/zfy_data/nuscenes/nuscenes_derived/lwlr_predicted_sub_0317")
    # print(count)
    gen_validtxt_for_structral("/home/zfy/radar-camera-fusion-depth/training/nuscenes_sub","/home/zfy/radar-camera-fusion-depth/validation/nuscenes_structral",500)