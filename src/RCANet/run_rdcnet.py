import sys
sys.path.append("/home/zfy/RCMDNet/src")
import argparse
from RDCNet.rdcnet_main import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model from checkpoint')
parser.add_argument('--image_path',
    type=str, required=True, help='Path to file that contains validation paths for images')
parser.add_argument('--radar_path',
    type=str, required=True, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--conf_ground_truth_path',
    type=str, default=None, help='Path to file that contains validation paths for ground truth')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input channels for the image')
parser.add_argument('--input_channels_radar_depth',
    type=int, default=2, help='Number of input channels for the depth')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')

# Network settings
#rcnet
parser.add_argument('--rcnet_img_encoder_type',
    nargs='+', type=str, default=['resnet34', 'batch_norm'], help='Encoder type')
parser.add_argument('--rcnet_dep_encoder_type',
    nargs='+', type=str, default=['resnet18', 'batch_norm'], help='Encoder type')
parser.add_argument('--rcnet_n_filters_encoder_image',
    nargs='+', type=int, default=[32, 64, 128, 128, 128], help='Number of filters per layer')
parser.add_argument('--rcnet_n_filters_encoder_depth',
    nargs='+', type=int, default=[32, 64, 128, 128, 128], help='Number of neurons per layer')
parser.add_argument('--rcnet_decoder_type',
    nargs='+', type=str, default=['multiscale', 'batch_norm'], help='Decoder type')
parser.add_argument('--rcnet_n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 64, 32, 16], help='Number of filters per layer')
parser.add_argument('--rcnet_fusion_type',
    type=str, default='add', help='Range of image intensities after normalization')
parser.add_argument('--rcnet_guidance',
    type=str, default=None, help='Range of image intensities after normalization')
parser.add_argument('--rcnet_guidance_layers',
    nargs='+',type=int, default=[], help='Range of image intensities after normalization')  
parser.add_argument('--rcnet_fusion_layers',
    nargs='+',type=str, default=[], help='Range of image intensities after normalization')   
parser.add_argument('--dropout_prob',
    type=float, default=0.0, help='dropout_prob')


# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='kaiming_uniform', help='Range of image intensities after normalization')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Range of image intensities after normalization')

#Loss
parser.add_argument('--w_positive_class',
    type=float, default=0.0, help='Weight of loss computed against feature map')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--save_outputs',
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
parser.add_argument('--response_thr',
    type=float, default=0.5, help='Max range of depths to evaluate')

args = parser.parse_args()

if __name__ == '__main__':

    run(
        restore_path=args.restore_path,
        image_path=args.image_path,
        radar_path=args.radar_path,
        conf_ground_truth_path=args.conf_ground_truth_path,
        # Input settings
        input_channels_image=args.input_channels_image,
        input_channels_radar_depth=args.input_channels_radar_depth,
        normalized_image_range=args.normalized_image_range,
        # Network settings
        #rcnet
        rcnet_img_encoder_type=args.rcnet_img_encoder_type,
        rcnet_dep_encoder_type=args.rcnet_dep_encoder_type,
        rcnet_n_filters_encoder_image=args.rcnet_n_filters_encoder_image, 
        rcnet_n_filters_encoder_depth=args.rcnet_n_filters_encoder_depth,
        rcnet_decoder_type=args.rcnet_decoder_type,
        rcnet_n_filters_decoder=args.rcnet_n_filters_decoder,
        rcnet_fusion_type=args.rcnet_fusion_type,
        rcnet_guidance=args.rcnet_guidance,
        rcnet_guidance_layers=args.rcnet_guidance_layers,
        rcnet_fusion_layers=args.rcnet_fusion_layers,
        dropout_prob=args.dropout_prob,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        w_positive_class=args.w_positive_class,
        # Output settings
        output_dirpath=args.output_dirpath,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        verbose=args.verbose,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        response_thr=args.response_thr)
