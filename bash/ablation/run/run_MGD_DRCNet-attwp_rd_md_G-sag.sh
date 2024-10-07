export CUDA_VISIBLE_DEVICES=3

python -u src/FusionNet/run_fusionnet.py \
--restore_path /home/zfy/RCMDNet/trained_fusionnet_ablation/0829222728__Lfunc_l1_wdenseL_1.0_wlidarL_2.0_wpercepL_0.0_wsmoothness_0.0_fusiontype_add_guidance_cbam_output_type_res_total_epoch_300--study_of_SAG_cbam_prelr_GPU_4/model-480000.pth \
--depths_in_use radar radar_dilation \
--image_path testing/nuscenes/nuscenes_test_dany_metric_predicted.txt \
--depth_path testing/nuscenes/nuscenes_test_radar_image.txt \
--radar_dilation_path testing/nuscenes/nuscenes_test_selected_dilation_radar_attwp.txt \
--ground_truth_path testing/nuscenes/nuscenes_test_lidar.txt \
--input_channels_image 1 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--img_encoder_type resnet34 batch_norm \
--dep_encoder_type resnet18 batch_norm \
--n_filters_encoder_image 32 64 128 256 256 256 \
--n_filters_encoder_depth 16 32 64 128 128 128 \
--fusion_type add \
--fusion_layers 1 2 3 4 5 6 \
--guidance cbam \
--guidance_layers 3 4 5 6 \
--decoder_type multiscale batch_norm \
--output_type res \
--dropout_prob 0.0 \
--n_filters_decoder 256 256 128 64 64 32 \
--n_resolutions_decoder 1 \
--min_predict_depth 1.0 \
--max_predict_depth 100.0 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--output_dirpath /data/zfy_data/nuscenes/virl_for_paper/output_depth/ours \
--keep_input_filenames \
--verbose \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 50.0 \
