export CUDA_VISIBLE_DEVICES=2

python -u src/RDCNet/run_rdcnet.py \
--restore_path trained_rdcnet/0812225515__Input_rd--attention_wp_adabins_GPU_5/model-343000.pth \
--image_path testing/nuscenes/nuscenes_test_image.txt \
--radar_path testing/nuscenes/nuscenes_test_radar_dilation_adabins.txt \
--conf_ground_truth_path testing/nuscenes/nuscenes_test_conf_gt_adabins.txt \
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
--w_positive_class 10 \
--response_thr 0.5 \
--output_dirpath /data/zfy_data/nuscenes/nuscenes_derived_test/rdcnet_adabins \
--keep_input_filenames \
--verbose \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 80.0 \
--save_outputs \
