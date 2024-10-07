export CUDA_VISIBLE_DEVICES=3

python -u src/FusionNet/train_fusionnet.py \
--depths_in_use radar radar_dilation \
--train_image_path training/nuscenes/nuscenes_train_dany_metric_predicted.txt \
--train_depth_path training/nuscenes/nuscenes_train_radar_image.txt \
--train_radar_dilation_path training/nuscenes/nuscenes_train_radar_dilation.txt \
--train_ground_truth_path training/nuscenes/nuscenes_train_ground_truth_interp.txt \
--train_lidar_map_path training/nuscenes/nuscenes_train_ground_truth.txt \
--val_image_path testing/nuscenes/nuscenes_test_dany_metric_predicted.txt \
--val_depth_path testing/nuscenes/nuscenes_test_radar_image.txt \
--val_radar_dilation_path testing/nuscenes/nuscenes_test_radar_dilation.txt \
--val_ground_truth_path testing/nuscenes/nuscenes_test_lidar.txt \
--batch_size 16 \
--n_height 448 \
--n_width 448 \
--input_channels_image 1 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--img_encoder_type resnet34 batch_norm \
--dep_encoder_type resnet18 batch_norm \
--n_filters_encoder_image 32 64 128 256 256 256 \
--n_filters_encoder_depth 16 32 64 128 128 128 \
--fusion_type add \
--fusion_layers 1 2 3 4 5 6 \
--decoder_type multiscale batch_norm \
--output_type res \
--n_filters_decoder 256 256 128 64 64 32 \
--n_resolutions_decoder 1 \
--min_predict_depth 1.0 \
--max_predict_depth 100.0 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--learning_rates 1e-3 5e-4 1e-4 \
--learning_schedule 200 250 300 \
--loss_func l1 \
--w_smoothness 0.0 \
--w_lidar_loss 2.0 \
--w_dense_loss 1.0 \
--w_weight_decay 0.0 \
--loss_smoothness_kernel_size -1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--ground_truth_dilation_kernel_size -1 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--augmentation_random_flip_type horizontal \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 80.0 \
--resultsave_dirpath trained_fusionnet \
--n_step_per_checkpoint 20000 \
--n_step_per_summary 5000 \
--n_step_per_validation 110000 \
--start_validation_step 100000 \
--n_thread 8 \
--disc MGD_rd_md \
--seed 40