export CUDA_VISIBLE_DEVICES=4

python -u src/RDCNet/train_rdcnet.py \
--train_image_path training/nuscenes/nuscenes_train_image.txt \
--train_radar_path training/nuscenes/nuscenes_train_radar_dilation_adabins.txt \
--train_conf_ground_truth_path training/nuscenes/nuscenes_train_conf_gt.txt \
--val_image_path testing/nuscenes/nuscenes_test_image.txt \
--val_radar_path testing/nuscenes/nuscenes_test_radar_dilation.txt \
--val_conf_ground_truth_path training/nuscenes/nuscenes_train_conf_gt_adabins.txt \
--batch_size 6 \
--n_height 352 \
--n_width 704 \
--input_channels_image 3 \
--input_channels_radar_depth 1 \
--normalized_image_range 0 1 \
--rcnet_img_encoder_type resnet34 batch_norm \
--rcnet_dep_encoder_type resnet18 batch_norm \
--rcnet_n_filters_encoder_image 32 64 128 256 256 256 \
--rcnet_n_filters_encoder_depth 16 32 64 128 128 128 \
--rcnet_fusion_type attention_wp \
--rcnet_fusion_layers weight_and_project weight_and_project weight_and_project weight_and_project weight_and_project attention \
--rcnet_decoder_type multiscale batch_norm \
--rcnet_n_filters_decoder 256 256 128 64 64 32 \
--dropout_prob 0.0 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--learning_rates 3e-4 \
--learning_schedule 200 \
--w_dense_loss 1.0 \
--w_positive_class 10 \
--w_conf_loss 1 \
--w_weight_decay 0.0 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--augmentation_random_flip_type horizontal \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 80.0 \
--response_thr 0.5 \
--resultsave_dirpath trained_rdcnet \
--n_step_per_checkpoint 10000 \
--n_step_per_summary 5000 \
--n_step_per_validation 20000 \
--n_thread 8 \
--disc attention_wp_adabins \
--seed 40

