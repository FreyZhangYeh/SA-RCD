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


class FusionNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) depth
        (3) response
        (4) ground truth

    Arg(s):
        image_paths : list[str]
            paths to images
        depth_paths : list[str]
            paths to depth maps
        response_paths : list[str]
            paths to response maps
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        shape : list[int]
            height, width tuple for random crop
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, top, bottom, left, right, center
    '''

    def __init__(self,
                 image_paths,
                 depth_paths,
                 mono_depth_paths,
                 radar_dilation_paths,
                 response_paths,
                 ground_truth_paths,
                 lidar_map_paths,
                 shape=None,
                 random_crop_type=['none']):

        self.n_sample = len(image_paths)

        for paths in [depth_paths, mono_depth_paths, radar_dilation_paths, response_paths, ground_truth_paths, lidar_map_paths]:
            assert paths is None or self.n_sample == len(paths)

        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.mono_depth_paths = mono_depth_paths
        self.radar_dilation_paths = radar_dilation_paths
        self.response_paths = response_paths
        self.ground_truth_paths = ground_truth_paths
        self.lidar_map_paths = lidar_map_paths

        self.shape = shape

        self.do_random_crop = \
            self.shape is not None and all([x > 0 for x in self.shape])

        # Augmentation
        self.random_crop_type = random_crop_type

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format) if 'nuscenes_origin' in self.image_paths[index] else data_utils.load_depth(
                self.image_paths[index],
                data_format=self.data_format)

        # Load depth
        depth = data_utils.load_depth(
            self.depth_paths[index],
            data_format=self.data_format) if self.depth_paths is not None else None
        
        mono_depth = data_utils.load_depth(
            self. mono_depth_paths[index],
            data_format=self.data_format) if self.mono_depth_paths is not None else None
        
        radar_dilation = data_utils.load_depth(
            self.radar_dilation_paths[index],
            data_format=self.data_format) if self.radar_dilation_paths is not None else None


        # Load response
        response = data_utils.load_depth(
            self.response_paths[index],
            data_format=self.data_format) if self.response_paths is not None else None

        # Load ground truth depth
        ground_truth = data_utils.load_depth(
            self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load lidar map depth
        lidar_map = data_utils.load_depth(
            self.lidar_map_paths[index],
            data_format=self.data_format)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image, depth,  mono_depth, radar_dilation, response, ground_truth, lidar_map] = random_crop(
                inputs=[image, depth, mono_depth, radar_dilation, response, ground_truth, lidar_map],
                shape=self.shape,
                crop_type=self.random_crop_type)

        # Convert to float32
        image, depth, mono_depth, radar_dilation, response, ground_truth, lidar_map = [
            T.astype(np.float32) if T is not None else T
            for T in [image, depth, mono_depth, radar_dilation, response, ground_truth, lidar_map]
        ]

        inputs = {"image":image, 
                  "ground_truth":ground_truth, 
                  "lidar_map":lidar_map}
        
        if depth is not None:
            inputs["radar_depth"] = depth
        if response is not None:
            inputs["response"] = response
        if mono_depth is not None:
            inputs["mono_depth"] = mono_depth
        if radar_dilation is not None:
            inputs["radar_dilation"] = radar_dilation

        return inputs


    def __len__(self):
        return self.n_sample


class FusionNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) depth
        (3) response
        (4) ground truth (if available)

    Arg(s):
        image_paths : list[str]
            paths to images
        depth_paths : list[str]
            paths to depth maps
        response_paths : list[str]
            paths to response maps
        ground_truth_paths : list[str]
            paths to ground truth depth maps
    '''

    def __init__(self,
                 image_paths,
                 depth_paths,
                 mono_depth_paths,
                 radar_dilation_paths,
                 response_paths,
                 ground_truth_paths):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.mono_depth_paths = mono_depth_paths if mono_depth_paths is not None and None not in mono_depth_paths else None
        self.radar_dilation_paths = radar_dilation_paths if radar_dilation_paths is not None and None not in radar_dilation_paths else None
        self.response_paths = response_paths if response_paths is not None and None not in response_paths else None

        if ground_truth_paths is not None and None not in ground_truth_paths:
            assert self.n_sample == len(ground_truth_paths)
            self.ground_truth_available = True
        else:
            self.ground_truth_available = False

        self.ground_truth_paths = ground_truth_paths

        for paths in [depth_paths, response_paths, ground_truth_paths,mono_depth_paths,radar_dilation_paths]:
            assert paths is None or self.n_sample == len(paths)

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format) if 'nuscenes_origin' in self.image_paths[index] else data_utils.load_depth(
                self.image_paths[index],
                data_format=self.data_format)

        # Load depth
        depth = data_utils.load_depth(
            self.depth_paths[index],
            data_format=self.data_format) if self.depth_paths is not None else None
        
        mono_depth = data_utils.load_depth(
            self.mono_depth_paths[index],
            data_format=self.data_format) if self.mono_depth_paths is not None else None
        
        radar_dilation = data_utils.load_depth(
            self.radar_dilation_paths[index],
            data_format=self.data_format) if self.radar_dilation_paths is not None else None

        # Load response
        response = data_utils.load_depth(
            self.response_paths[index],
            data_format=self.data_format) if self.response_paths is not None else None
        
        if self.ground_truth_available:
            # Load ground truth depth
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)


        inputs = {"image":image, 
                  "ground_truth":ground_truth}

        image, depth, mono_depth, radar_dilation, response, ground_truth = [
            T.astype(np.float32) if T is not None else T
            for T in [image, depth, mono_depth, radar_dilation, response, ground_truth]
        ]

        if depth is not None:
            inputs["radar_depth"] = depth 
        if response is not None:
            inputs["response"] = response
        if mono_depth is not None:
            inputs["mono_depth"] = mono_depth
        if radar_dilation is not None:
            inputs["radar_dilation"] = radar_dilation

        return inputs

    def __len__(self):
        return self.n_sample
