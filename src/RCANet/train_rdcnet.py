import sys
sys.path.append("/home/zfy/RCMDNet/src")
import argparse
from RDCNet.rdcnet_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--depths_in_use',
    nargs='+', type=str, required=False, default=["depth"],help='depths use for fusionnet')
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to file that contains training paths for images')
parser.add_argument('--train_radar_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_radar_dilation_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_mono_depth_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_rela_depth_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_conf_ground_truth_path',
    type=str, required=True, help='Path to file that contains training paths for conf ground truth')
parser.add_argument('--val_image_path',
    type=str, required=True, help='Path to file that contains validation paths for images')
parser.add_argument('--val_radar_path',
    type=str, required=True, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--val_mono_depth_path',
    type=str, required=None, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--val_rela_depth_path',
    type=str, required=None, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--val_conf_ground_truth_path',
    type=str, required=None, help='Path to file that contains validation paths for ground truth')

#batch settings
parser.add_argument('--batch_size',
    type=int, default=64, help='Batch Size for the input')
parser.add_argument('--patch_size',
    nargs='+', type=int, default=[768, 288], help='Height, width of input patch')
parser.add_argument('--n_height',
    type=int, default=900, help='Height of the input')
parser.add_argument('--n_width',
    type=int, default=1600, help='Width of the input')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input channels for the image')
parser.add_argument('--input_channels_radar_depth',
    type=int, default=2, help='Number of input channels for the depth')
parser.add_argument('--input_channels_quasi_depth',
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

#public
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

# Training Settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[5e-5, 1e-4, 2e-4, 1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[2, 5, 10, 12, 15], help='Space delimited list to change learning rate')

# Loss settings
parser.add_argument('--loss_func',
    type=str, default='l1', help='Loss function type')
parser.add_argument('--w_smoothness',
    type=float, default=0.0, help='Weight of local smoothness loss')
parser.add_argument('--w_lidar_loss',
    type=float, default=0.0, help='Weight of loss computed against a single lidar map')
parser.add_argument('--w_dense_loss',
    type=float, default=0.0, help='Weight of loss computed against a dense depth map')
parser.add_argument('--w_perceptual_loss',
    type=float, default=0.0, help='Weight of loss computed against feature map')
parser.add_argument('--w_conf_loss',
    type=float, default=0.0, help='Weight of loss computed against feature map')
parser.add_argument('--w_quasi_loss',
    type=float, default=0.0, help='Weight of loss computed against feature map')
parser.add_argument('--w_weight_decay',
    type=float, default=0.0, help='Weight of weight decay')
parser.add_argument('--loss_smoothness_kernel_size',
    type=int, default=-1, help='Smoothness loss kernel size')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=-1, help='Outlier removal kernel size')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=-1, help='Threshold to consider a point an outlier')
parser.add_argument('--ground_truth_dilation_kernel_size',
    type=int, default=-1, help='Dilation kernel size')
parser.add_argument('--w_positive_class',
    type=float, default=1.0, help='Weight of positive class')
parser.add_argument('--max_distance_correspondence',
    type=float, default=0.4, help='Max distance to consider two points correspondence')
parser.add_argument('--set_invalid_to_negative_class',
    action='store_true', help='If set then any invalid locations are treated as negative class')

# Augmentation Settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['horizontal', 'vertical'], help='Random crop adjustment for data augmentation')
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[0.80, 1.20], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[0.80, 1.20], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[0.80, 1.20], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Flip type for augmentation: none, horizontal, vertical')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default=['none'], help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')



# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Min range of depths to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100, help='Max range of depths to evaluate')
parser.add_argument('--response_thr',
    type=float, default=0.5, help='Max range of depths to evaluate')


# Checkpoint and summary settings
# parser.add_argument('--checkpoint_dirpath',
#     type=str, required=True, help='Path to load checkpoints')
parser.add_argument('--resultsave_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=100, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=100, help='Number of samples to include in visual display summary')
parser.add_argument('--n_step_per_validation',
    type=int, default=100, help='Step to start performing validation')
parser.add_argument('--structralnet_restore_path',
    type=str, default=None, help='Path to restore structralnet model from checkpoint')
parser.add_argument('--rdcnet_restore_path',
    type=str, default=None, help='Path to restore rcmdnet from checkpoint')

# Hardware and debugging
parser.add_argument('--device',
    type=str, default='cuda', help='Device')
parser.add_argument('--n_thread',
    type=int, default=10, help='Number of threads for fetching')
parser.add_argument('--disc',
    type=str, default="baseline", help='Brief introduction of the exp')
parser.add_argument('--seed',
    type=int, default=355123027, help='Seed to reconstruct training precess')


args = parser.parse_args()

if __name__ == '__main__':

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    train(train_image_path=args.train_image_path,
          train_radar_path=args.train_radar_path,
          train_conf_ground_truth_path=args.train_conf_ground_truth_path,
          val_image_path=args.val_image_path,
          val_radar_path=args.val_radar_path,
          val_conf_ground_truth_path=args.val_conf_ground_truth_path,
          # Batch settings
          batch_size=args.batch_size,
          n_height=args.n_height,
          n_width=args.n_width,
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
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_saturation=args.augmentation_random_saturation,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          # Loss function settings
          w_weight_decay=args.w_weight_decay,
          w_dense_loss=args.w_dense_loss,
          w_conf_loss=args.w_conf_loss,
          w_positive_class=args.w_positive_class,
          set_invalid_to_negative_class=args.set_invalid_to_negative_class,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          response_thr=args.response_thr,
          # Checkpoint settings
          #checkpoint_dirpath=args.checkpoint_dirpath,
          resultsave_dirpath=args.resultsave_dirpath,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_step_per_validation=args.n_step_per_validation,
          rdcnet_restore_path=args.rdcnet_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread,
          disc=args.disc,
          seed=args.seed)
