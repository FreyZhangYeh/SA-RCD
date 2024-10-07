import argparse
from fusionnet_main import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--depths_in_use',
    nargs='+', type=str, required=False, default=["depth"],help='depths use for fusionnet')
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model from checkpoint')
parser.add_argument('--image_path',
    type=str, required=True, help='Path to file that contains validation paths for images')
parser.add_argument('--radar_dilation_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--depth_path',
    type=str, required=True, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--response_path',
    type=str, default=None, help='Path to file that contains validation paths for response maps')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to file that contains validation paths for ground truth')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input channels for the image')
parser.add_argument('--input_channels_depth',
    type=int, default=2, help='Number of input channels for the depth')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')

# Network settings
parser.add_argument('--img_encoder_type',
    nargs='+', type=str, default='resnet34', help='Type of encoder')
parser.add_argument('--dep_encoder_type',
    nargs='+', type=str, default='resnet18', help='Type of encoder')
parser.add_argument('--frozen_strategy',
    nargs='+', type=str, default=None, help='Parameters to be frozen')
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=[0, 1], help='Range of image intensities after normalization')
parser.add_argument('--n_filters_encoder_depth',
    nargs='+', type=int, default=[0, 1], help='Range of image intensities after normalization')
parser.add_argument('--fusion_type',
    type=str, default='add', help='Range of image intensities after normalization')
parser.add_argument('--guidance',
    type=str, default=None, help='Range of image intensities after normalization')
parser.add_argument('--fusion_layers',
    nargs='+',type=int, default=[1, 2], help='Range of image intensities after normalization')  
parser.add_argument('--guidance_layers',
    nargs='+',type=int, default=None, help='Range of image intensities after normalization')  
parser.add_argument('--decoder_type',
    nargs='+', type=str, default='multiscale', help='Range of image intensities after normalization')   
parser.add_argument('--output_type',
    type=str, default='metric_depth', help='Range of image intensities after normalization')  
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[0, 1], help='Range of image intensities after normalization')
parser.add_argument('--n_resolutions_decoder',
    type=int, default=0, help='Range of image intensities after normalization')
parser.add_argument('--dropout_prob',
    type=float, default=0.0, help='dropout_prob')
parser.add_argument('--min_predict_depth',
    type=float, default=0, help='Min range of depths to predict')
parser.add_argument('--max_predict_depth',
    type=float, default=100, help='Max range of depths to predict')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='kaiming_uniform', help='Range of image intensities after normalization')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Range of image intensities after normalization')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then save outputs to output directory')
parser.add_argument('--save_dnzs_path',
    action='store_true', help='If set then save outputs to output directory')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original file names')
parser.add_argument('--verbose',
    action='store_true', help='If set then print progress')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Min range of depths to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100, help='Max range of depths to evaluate')


args = parser.parse_args()

if __name__ == '__main__':

    run(restore_path=args.restore_path,
        depths_in_use=args.depths_in_use,
        image_path=args.image_path,
        depth_path=args.depth_path,
        radar_dilation_path=args.radar_dilation_path,
        response_path=args.response_path,
        ground_truth_path=args.ground_truth_path,
        # Input settings
        input_channels_image=args.input_channels_image,
        input_channels_depth=args.input_channels_depth,
        normalized_image_range=args.normalized_image_range,
        # Network settings
        img_encoder_type=args.img_encoder_type,
        dep_encoder_type=args.dep_encoder_type,
        n_filters_encoder_image=args.n_filters_encoder_image,
        n_filters_encoder_depth=args.n_filters_encoder_depth,
        fusion_type=args.fusion_type,
        guidance=args.guidance,
        guidance_layers=args.guidance_layers,
        fusion_layers=args.fusion_layers,
        decoder_type=args.decoder_type,
        output_type=args.output_type,
        n_filters_decoder=args.n_filters_decoder,
        n_resolutions_decoder=args.n_resolutions_decoder,
        dropout_prob=args.dropout_prob,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        # Output settings
        output_dirpath=args.output_dirpath,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        verbose=args.verbose,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth)
