import torch
import numpy as np
from utils import data_utils
import random


def random_sample(T):
    '''
    Arg(s):
        T : numpy[float32]
            C x N array
    Returns:
        numpy[float32] : random sample from T
    '''

    index = np.random.randint(0, T.shape[0])
    return T[index, :]

def random_crop(inputs, shape, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        crop_type : str
            none, horizontal, vertical, anchored, top, bottom, left, right, center
    Return:
        list[numpy[float32]] : list of cropped inputs
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width

    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    # If left alignment, then set starting height to 0
    if 'left' in crop_type:
        x_start = 0

    # If right alignment, then set starting height to right most position
    elif 'right' in crop_type:
        x_start = d_width

    elif 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If top alignment, then set starting height to 0
    if 'top' in crop_type:
        y_start = 0

    # If bottom alignment, then set starting height to lowest position
    elif 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    elif 'center' in crop_type:
        pass

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width

    outputs = [
        T[:, y_start:y_end, x_start:x_end] if T is not None else T for T in inputs
    ]

    return outputs

def crop_radar_points(radar_points, x_start, x_end, y_start, y_end):
    """
    Crop the radar points based on the image cropping parameters.
    
    Args:
        radar_points (numpy.ndarray): Array of shape [k, 3], where each row is (x, y, d).
        x_start (int): Start coordinate of the crop in x direction.
        x_end (int): End coordinate of the crop in x direction.
        y_start (int): Start coordinate of the crop in y direction.
        y_end (int): End coordinate of the crop in y direction.
        
    Returns:
        numpy.ndarray: Cropped radar points within the image crop.
    """
    # Filtering points within the cropping region
    mask = (radar_points[:, 0] >= x_start) & (radar_points[:, 0] < x_end) & \
           (radar_points[:, 1] >= y_start) & (radar_points[:, 1] < y_end)
    radar_points_crop = radar_points[mask]

    # Adjusting the coordinates to the new cropped image coordinates
    radar_points_crop[:, 0] -= x_start
    radar_points_crop[:, 1] -= y_start
    
    return radar_points_crop
   

class RDCNetTrainingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 #"rcnet"
                 image_paths,
                 radar_paths,
                 conf_ground_truth_paths,
                 shape=None,
                 random_crop_type=['none']
                 ):
        #amount check
        self.n_sample = len(image_paths)
        for paths in [radar_paths,conf_ground_truth_paths]:
            assert paths is None or self.n_sample == len(paths)

        #rcnet self
        self.image_paths = image_paths
        self.radar_paths = radar_paths
        self.conf_ground_truth_paths = conf_ground_truth_paths

        #fusionnet self
        self.shape = shape
        self.do_random_crop = \
            self.shape is not None and all([x > 0 for x in self.shape])
        self.random_crop_type = random_crop_type
        self.data_format = 'CHW'


    def __getitem__(self, index):

        #######################FusionNet############################
        # Load image
        image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format) if 'nuscenes_origin' in self.image_paths[index] else data_utils.load_depth(
                self.image_paths[index],
                data_format=self.data_format)
        
        radar_depth = data_utils.load_depth(
            self.radar_paths[index],
            data_format=self.data_format)
        
        conf_ground_truth = data_utils.load_depth(
            self.conf_ground_truth_paths[index],
            data_format=self.data_format
        )
        
        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image, radar_depth, conf_ground_truth] = random_crop(
                inputs=[image, radar_depth, conf_ground_truth],
                shape=self.shape,
                crop_type=self.random_crop_type)
        
        # Convert to float32
        image, radar_depth, conf_ground_truth = [
            T.astype(np.float32) if T is not None else T
            for T in [image, radar_depth, conf_ground_truth]
        ]

        
        rcnet_inputs = {
                "image":image,
                "radar_depth":radar_depth, 
                "conf_ground_truth":conf_ground_truth,
                    }
        

        return rcnet_inputs
        
    def __len__(self):
        return self.n_sample

class RDCNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) radar points
        (3) ground truth (if available)

    Arg(s):
        image_paths : list[str]
            paths to images
        radar_paths : list[str]
            paths to radar points
        ground_truth_paths : list[str]
            paths to ground truth paths
    '''

    def __init__(self, 
                 image_paths, 
                 radar_paths, 
                 conf_ground_truth_paths=None,
                ):

        self.n_sample = len(image_paths)

        assert self.n_sample == len(radar_paths)

        self.image_paths = image_paths
        self.radar_paths = radar_paths

        self.data_format = 'CHW'
        self.conf_ground_truth_paths = conf_ground_truth_paths

        for paths in [radar_paths, conf_ground_truth_paths]:
            assert paths is None or self.n_sample == len(paths)

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format) if 'nuscenes_origin' in self.image_paths[index] else data_utils.load_depth(
                self.image_paths[index],
                data_format=self.data_format)

        # Load radar points N x 3
        radar_depth = data_utils.load_depth(
            self.radar_paths[index],
            data_format=self.data_format)
        
        conf_ground_truth =  data_utils.load_depth(
            self.conf_ground_truth_paths[index],
            data_format=self.data_format) if self.conf_ground_truth_paths is not None else None

        # Convert to float32
        image, radar_depth, conf_ground_truth = [
                T.astype(np.float32) if T is not None else T
                for T in [image, radar_depth, conf_ground_truth]
            ]
        
        rcnet_inputs = {"image":image,
                        "radar_depth":radar_depth,
                        }
        
        if conf_ground_truth is not None:
            rcnet_inputs["conf_ground_truth"] = conf_ground_truth

        return rcnet_inputs
    
    def __len__(self):
        return self.n_sample
