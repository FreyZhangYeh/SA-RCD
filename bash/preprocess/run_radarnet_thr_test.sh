export CUDA_VISIBLE_DEVICES=7

python -u src/run_radarnet.py \
--restore_path checkpoints/Radarnet/model-195000.pth \
--image_path testing/nuscenes/nuscenes_test_image.txt \
--radar_path testing/nuscenes/nuscenes_test_radar.txt \
--patch_size 900 288 \
--normalized_image_range 0 1 \
--encoder_type radarnetv1 batch_norm \
--n_filters_encoder_image 32 64 128 128 128 \
--n_neurons_encoder_depth 32 64 128 128 128 \
--decoder_type multiscale batch_norm \
--n_filters_decoder 256 128 64 32 16 \
--weight_initializer kaiming_uniform \
--activation_func leaky_relu \
--output_dirpath /data/zfy_data/nuscenes/nuscenes_derived_test \
--save_outputs \
--keep_input_filenames \
--verbose \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--response_thr 0.5
