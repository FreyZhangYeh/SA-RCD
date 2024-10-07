import argparse
from fusionnet_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--depths_in_use',
    nargs='+', type=str, required=False, default=["depth"],help='depths use for fusionnet')
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to file that contains training paths for images')
parser.add_argument('--train_depth_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_mono_depth_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_response_path',
    type=str, default=None, help='Path to file that contains training paths for response maps')
parser.add_argument('--train_radar_dilation_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Path to file that contains training paths for ground truth')
parser.add_argument('--train_lidar_map_path',
    type=str, required=True, help='Path to file that contains training paths for single lidar scan maps')
parser.add_argument('--val_image_path',
    type=str, required=True, help='Path to file that contains validation paths for images')
parser.add_argument('--val_depth_path',
    type=str, default=None, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--val_mono_depth_path',
    type=str, default=None, help='Path to file that contains training paths for depth maps')
parser.add_argument('--val_response_path',
    type=str, default=None, help='Path to file that contains validation paths for response maps')
parser.add_argument('--val_radar_dilation_path',
    type=str, default=None, help='Path to file that contains validation paths for depth maps')
parser.add_argument('--val_ground_truth_path',
    type=str, required=True, help='Path to file that contains validation paths for ground truth')

parser.add_argument('--batch_size',
    type=int, default=64, help='Batch Size for the input')
parser.add_argument('--n_height',
    type=int, default=900, help='Height of the input')
parser.add_argument('--n_width',
    type=int, default=1600, help='Width of the input')

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

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0, help='Min range of depths to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100, help='Max range of depths to evaluate')

# Checkpoint and summary settings
# parser.add_argument('--checkpoint_dirpath',
#     type=str, required=True, help='Path to load checkpoints')
parser.add_argument('--resultsave_dirpath',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=100, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=100, help='Number of samples to include in visual display summary')
parser.add_argument('--start_validation_step',
    type=int, default=40000, help='Number of samples to include in visual display summary')
parser.add_argument('--n_step_per_validation',
    type=int, default=100, help='Step to start performing validation')
parser.add_argument('--transfer_type',
    nargs='+', type=str, default=None, help='network tranfered to fusionnet')
parser.add_argument('--structralnet_restore_path',
    type=str, default=None, help='Path to restore structralnet model from checkpoint')
parser.add_argument('--radar_camera_fusionnet_restore_path',
    type=str, default=None, help='Path to restore radar_camera_fusionnet model from checkpoint')

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

    train(depths_in_use=args.depths_in_use,
          train_image_path=args.train_image_path,
          train_depth_path=args.train_depth_path,
          train_mono_depth_path=args.train_mono_depth_path,
          train_radar_dilation_path=args.train_radar_dilation_path,
          train_response_path=args.train_response_path,
          train_ground_truth_path=args.train_ground_truth_path,
          train_lidar_map_path=args.train_lidar_map_path,
          val_image_path=args.val_image_path,
          val_depth_path=args.val_depth_path,
          val_mono_depth_path=args.val_mono_depth_path,
          val_radar_dilation_path=args.val_radar_dilation_path,
          val_response_path=args.val_response_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Batch settings
          batch_size=args.batch_size,
          n_height=args.n_height,
          n_width=args.n_width,
          # Input settings
          input_channels_image=args.input_channels_image,
          input_channels_depth=args.input_channels_depth,
          normalized_image_range=args.normalized_image_range,
          # Network settings
          img_encoder_type=args.img_encoder_type,
          dep_encoder_type=args.dep_encoder_type,
          frozen_strategy=args.frozen_strategy,
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
          loss_func=args.loss_func,
          w_smoothness=args.w_smoothness,
          w_weight_decay=args.w_weight_decay,
          loss_smoothness_kernel_size=args.loss_smoothness_kernel_size,
          w_lidar_loss=args.w_lidar_loss,
          w_dense_loss=args.w_dense_loss,
          w_perceptual_loss=args.w_perceptual_loss,
          ground_truth_outlier_removal_kernel_size=args.outlier_removal_kernel_size,
          ground_truth_outlier_removal_threshold=args.outlier_removal_threshold,
          ground_truth_dilation_kernel_size=args.ground_truth_dilation_kernel_size,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          #checkpoint_dirpath=args.checkpoint_dirpath,
          resultsave_dirpath=args.resultsave_dirpath,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          start_validation_step=args.start_validation_step,
          n_step_per_validation=args.n_step_per_validation,
          transfer_type=args.transfer_type,
          structralnet_restore_path = args.structralnet_restore_path,
          radar_camera_fusionnet_restore_path=args.radar_camera_fusionnet_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread,
          disc=args.disc,
          seed=args.seed)
