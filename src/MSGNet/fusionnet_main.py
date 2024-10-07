import sys
sys.path.append("/home/zfy/RCMDNet/src")
import os, time
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from utils import data_utils, eval_utils
from datasets import datasets_fusionnet as datasets
from utils.log_utils import log
from fusionnet_model import FusionNetModel
from fusionnet_transforms import Transforms
from utils.net_utils import OutlierRemoval
from tqdm import tqdm
from datetime import datetime
import random
from utils.misc import colorize
from utils.data_utils import hist
from utils.eval_utils import apply_thr

def train(depths_in_use,
          train_image_path,
          train_depth_path,
          train_mono_depth_path,
          train_radar_dilation_path,
          train_response_path,
          train_ground_truth_path,
          train_lidar_map_path,
          val_image_path,
          val_depth_path,
          val_mono_depth_path,
          val_radar_dilation_path,
          val_response_path,
          val_ground_truth_path,
          # Batch settings
          batch_size,
          n_height,
          n_width,
          # Input settings
          input_channels_image,
          input_channels_depth,
          normalized_image_range,
          # Network settings
          img_encoder_type,
          dep_encoder_type,
          frozen_strategy,
          n_filters_encoder_image,
          n_filters_encoder_depth,
          fusion_type,
          guidance_layers,
          fusion_layers,
          guidance,
          decoder_type,
          output_type,
          n_filters_decoder,
          n_resolutions_decoder,
          dropout_prob,
          min_predict_depth,
          max_predict_depth,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_crop_type,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          augmentation_random_flip_type,
          # Loss settings
          loss_func,
          w_smoothness,
          w_weight_decay,
          loss_smoothness_kernel_size,
          w_lidar_loss,
          w_dense_loss,
          w_perceptual_loss,
          ground_truth_outlier_removal_kernel_size,
          ground_truth_outlier_removal_threshold,
          ground_truth_dilation_kernel_size,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          #checkpoint_dirpath,
          resultsave_dirpath,
          n_step_per_summary,
          n_step_per_checkpoint,
          start_validation_step,
          n_step_per_validation,
          transfer_type,
          structralnet_restore_path,
          radar_camera_fusionnet_restore_path,
          # Hardware settings
          device,
          n_thread,
          disc,
          seed):
    
    # Minimize randomness
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = datetime.now()
    formatted_time = now.strftime("%m%d%H%M%S")
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    save_dir_info = "_Lfunc_" + loss_func + "_wdenseL_" + str(w_dense_loss) + "_wlidarL_" + str(w_lidar_loss) + \
                    "_wpercepL_" + str(w_perceptual_loss) + "_wsmoothness_" + str(w_smoothness) + "_fusiontype_" + fusion_type
    
    if transfer_type != None and transfer_type != "":
        save_dir_info += "_transfer_type_"
        if isinstance(transfer_type, list):
            for i in transfer_type:
                save_dir_info += i
        else:
            save_dir_info += transfer_type
  
    save_dir_info = save_dir_info + "_frozen_strategy_" + str(frozen_strategy)[1:-2] + "_" if frozen_strategy is not None else save_dir_info
    save_dir_info = save_dir_info + "_guidance_" + guidance if guidance is not None else save_dir_info 
    save_dir_info = save_dir_info + "_output_type_" + output_type if output_type is not None else save_dir_info
    #save_dir_info = save_dir_info + "_layers_" + str(guidance_layers) if guidance is not None else save_dir_info
    save_dir_info = save_dir_info + "_total_epoch_" + str(learning_schedule[-1])
    resultsave_dirpath = os.path.join(resultsave_dirpath,formatted_time + "_" + save_dir_info +  "--"  + disc + "_GPU_" + cuda_visible_devices)


    if not os.path.exists(resultsave_dirpath):
        os.makedirs(resultsave_dirpath)

    # Set up checkpoint and event paths
    depth_model_checkpoint_path = os.path.join(resultsave_dirpath, 'model-{}.pth')
    log_path = os.path.join(resultsave_dirpath, 'results.txt')
    event_path = os.path.join(resultsave_dirpath, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    train_image_paths = data_utils.read_paths(train_image_path)
    train_depth_paths = data_utils.read_paths(train_depth_path) if train_depth_path is not None else None
    train_mono_depth_paths = data_utils.read_paths(train_mono_depth_path) if train_mono_depth_path is not None else None
    train_radar_dilation_paths = data_utils.read_paths(train_radar_dilation_path) if train_radar_dilation_path is not None else None
    train_response_paths = data_utils.read_paths(train_response_path) if train_response_path is not None else None
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)
    train_lidar_map_paths = data_utils.read_paths(train_lidar_map_path)

    n_train_sample = len(train_image_paths)

    for paths in [train_depth_paths, train_response_paths, train_mono_depth_paths, train_radar_dilation_paths, train_ground_truth_paths, train_lidar_map_paths]:
        assert paths is None or n_train_sample == len(paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    # Set up data loader and data transforms
    train_dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetTrainingDataset(
            image_paths=train_image_paths,
            depth_paths=train_depth_paths,
            response_paths=train_response_paths,
            mono_depth_paths=train_mono_depth_paths,
            radar_dilation_paths=train_radar_dilation_paths,
            ground_truth_paths=train_ground_truth_paths,
            lidar_map_paths=train_lidar_map_paths,
            shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_thread)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation,
        random_flip_type=augmentation_random_flip_type)

    '''
    Set up paths for validation
    '''
    val_image_paths = data_utils.read_paths(val_image_path)
    val_depth_paths = data_utils.read_paths(val_depth_path) if val_depth_path is not None else None
    val_mono_depth_paths = data_utils.read_paths(val_mono_depth_path) if val_mono_depth_path is not None else None
    val_radar_dilation_paths = data_utils.read_paths(val_radar_dilation_path) if val_radar_dilation_path is not None else None
    val_response_paths = data_utils.read_paths(val_response_path) if train_response_path is not None else None
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    n_val_sample = len(val_image_paths)

    for paths in [val_depth_paths, val_response_paths, val_ground_truth_paths]:
        assert paths is None or n_val_sample == len(paths)

    val_dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetInferenceDataset(
            image_paths=val_image_paths,
            depth_paths=val_depth_paths,
            mono_depth_paths=val_mono_depth_paths,
            radar_dilation_paths=val_radar_dilation_paths,
            response_paths=val_response_paths,
            ground_truth_paths=val_ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    val_transforms = Transforms(
        normalized_image_range=normalized_image_range)

    # Initialize ground truth outlier removal
    if ground_truth_outlier_removal_kernel_size > 1 and ground_truth_outlier_removal_threshold > 0:
        ground_truth_outlier_removal = OutlierRemoval(
            kernel_size=ground_truth_outlier_removal_kernel_size,
            threshold=ground_truth_outlier_removal_threshold)
    else:
        ground_truth_outlier_removal = None

    # Initialize ground truth dilation
    if ground_truth_dilation_kernel_size > 1:
        ground_truth_dilation = torch.nn.MaxPool2d(
            kernel_size=ground_truth_dilation_kernel_size,
            stride=1,
            padding=ground_truth_dilation_kernel_size // 2)
    else:
        ground_truth_dilation = None

    '''
    Set up the model
    '''
    # Build network
    fusionnet_model = FusionNetModel(
        depths_in_use=depths_in_use,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        img_encoder_type=img_encoder_type,
        dep_encoder_type=dep_encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        guidance = guidance,
        guidance_layers = guidance_layers,
        fusion_layers = fusion_layers,
        decoder_type=decoder_type,
        n_resolution_decoder=n_resolutions_decoder,
        n_filters_decoder=n_filters_decoder,
        deconv_type='up',
        output_type=output_type,
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        dropout_prob=dropout_prob,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    fusionnet_model.to(device)
    #fusionnet_model.data_parallel()

    if len(cuda_visible_devices) >1:
        fusionnet_model.data_parallel()

    parameters_fusionnet_model = fusionnet_model.parameters()

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_image_path,
        train_depth_path,
        train_mono_depth_path,
        train_radar_dilation_path,
        train_response_path,
        train_ground_truth_path
    ]
    for path in train_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_depth_path,
        val_mono_depth_path,
        val_radar_dilation_path,
        val_response_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        depths_in_use,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        img_encoder_type=img_encoder_type,
        dep_encoder_type=dep_encoder_type,
        frozen_strategy=frozen_strategy,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        guidance=guidance,
        guidance_layers=guidance_layers,
        fusion_layers=fusion_layers,
        decoder_type=decoder_type,
        output_type=output_type,
        n_filters_decoder=n_filters_decoder,
        n_resolutions_decoder=n_resolutions_decoder,
        dropout_prob=dropout_prob,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_fusionnet_model)

    log_training_settings(
        log_path,
        # Training settings
        batch_size=batch_size,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_flip_type=augmentation_random_flip_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        loss_func=loss_func,
        w_smoothness=w_smoothness,
        w_weight_decay=w_weight_decay,
        w_lidar_loss=w_lidar_loss,
        w_dense_loss=w_dense_loss,
        w_perceptual_loss=w_perceptual_loss,
        loss_smoothness_kernel_size=loss_smoothness_kernel_size,
        outlier_removal_kernel_size=ground_truth_outlier_removal_kernel_size,
        outlier_removal_threshold=ground_truth_outlier_removal_threshold,
        ground_truth_dilation_kernel_size=ground_truth_dilation_kernel_size)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=resultsave_dirpath,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        n_step_per_validation=n_step_per_validation,
        start_validation_step=start_validation_step,
        structralnet_restore_path=structralnet_restore_path,
        transfer_type=transfer_type,
        radar_camera_fusionnet_restore_path=radar_camera_fusionnet_restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    # Initialize optimizer with starting learning rate
    optimizer = torch.optim.Adam([
        {
            'params': parameters_fusionnet_model,
            'weight_decay': w_weight_decay
        }],
        lr=learning_rates[0]  # 初始学习率
    )

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    # Start training
    train_step = 0

    if radar_camera_fusionnet_restore_path is not None and radar_camera_fusionnet_restore_path != '':
        train_step_retored, optimizer = fusionnet_model.restore_model(
            radar_camera_fusionnet_restore_path,
            optimizer=optimizer)
        
        n_train_step += train_step_retored
        train_step = train_step_retored

        # lr_scheduler.last_epoch = train_step
        # lr_scheduler.total_steps = int(n_train_step)
        
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rates[0],  # 使用最大学习率
        total_steps=int(n_train_step)+10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear',
        last_epoch=-1 if train_step == 0 else train_step,
        three_phase=False,
    )

    time_start = time.time()
    
    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        pbar = tqdm(total=n_train_sample)

        for batch_data in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            for key, in_ in batch_data.items():
                batch_data[key] = in_.to(device)

            # Apply augmentations and data transforms
            if "radar" in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" in depths_in_use:
                # Unpack data
                image, radar_depth, radar_dilation, response, ground_truth, lidar_map = (
                    batch_data["image"], batch_data["radar_depth"],
                    batch_data["radar_dilation"], batch_data["response"],
                    batch_data["ground_truth"], batch_data["lidar_map"]
                )
                
                radar_dilation = apply_thr(quasi_depth=radar_dilation,conf_map=response,thr=0.4)

                # Validate and preprocess depth
                radar_depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
                radar_depth_valid = radar_depth_valid.bool()

                radar_dilation_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                radar_dilation_valid = radar_dilation_valid.bool()

                radar_depth[~radar_depth_valid] = 0.0  # set invalid depth
                radar_dilation[~radar_dilation_valid] = 0.0

                # Apply transformations
                [image], [radar_depth, radar_dilation, response, ground_truth, lidar_map] = train_transforms.transform(
                    images_arr=[image],
                    range_maps_arr=[radar_depth, radar_dilation, response, ground_truth, lidar_map],
                    random_transform_probability=augmentation_probability
                )

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_depth"] = radar_depth
                batch_data["radar_dilation"] = radar_dilation
                batch_data["response"] = response
                batch_data["ground_truth"] = ground_truth
                batch_data["lidar_map"] = lidar_map

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" not in depths_in_use:
                response = None
                image, radar_depth, radar_dilation,  ground_truth, lidar_map = (
                    batch_data["image"], batch_data["radar_depth"],
                    batch_data["radar_dilation"], 
                    batch_data["ground_truth"], batch_data["lidar_map"]
                )

                # Validate and preprocess depth
                radar_depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
                radar_depth_valid = radar_depth_valid.bool()

                radar_dilation_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                radar_dilation_valid = radar_dilation_valid.bool()

                radar_depth[~radar_depth_valid] = 0.0  # set invalid depth
                radar_dilation[~radar_dilation_valid] = 0.0

                # Apply transformations
                [image], [radar_depth, radar_dilation, ground_truth, lidar_map] = train_transforms.transform(
                    images_arr=[image],
                    range_maps_arr=[radar_depth, radar_dilation,  ground_truth, lidar_map],
                    random_transform_probability=augmentation_probability
                )

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_depth"] = radar_depth
                batch_data["radar_dilation"] = radar_dilation
                batch_data["ground_truth"] = ground_truth
                batch_data["lidar_map"] = lidar_map

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" in depths_in_use and "radar_dilation" not in depths_in_use and "conf_map" not in depths_in_use:
                response = None; radar_dilation=None
                image, radar_depth,  ground_truth, lidar_map = (
                    batch_data["image"], batch_data["radar_depth"], batch_data["ground_truth"],batch_data["lidar_map"]
                )

                depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
                depth_valid = depth_valid.bool()
                
                radar_depth[~depth_valid] = 0.0  # set invalid dep

                # Apply transformations
                [image], [radar_depth, ground_truth, lidar_map] = train_transforms.transform(
                    images_arr=[image],
                    range_maps_arr=[radar_depth, ground_truth, lidar_map],
                    random_transform_probability=augmentation_probability
                )

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_depth"] = radar_depth
                batch_data["ground_truth"] = ground_truth
                batch_data["lidar_map"] = lidar_map

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" not in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" not in depths_in_use:
                radar_depth= None; response = None
                image, radar_dilation,  ground_truth, lidar_map = (
                    batch_data["image"], batch_data["radar_dilation"], batch_data["ground_truth"],batch_data["lidar_map"]
                )

                depth_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                depth_valid = depth_valid.bool()

                radar_dilation[~depth_valid] = 0.0  # set invalid dep

                # Apply transformations
                [image], [radar_dilation, ground_truth, lidar_map] = train_transforms.transform(
                    images_arr=[image],
                    range_maps_arr=[radar_dilation, ground_truth, lidar_map],
                    random_transform_probability=augmentation_probability
                )

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_dilation"] = radar_dilation
                batch_data["ground_truth"] = ground_truth
                batch_data["lidar_map"] = lidar_map

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" not in depths_in_use and "radar_dilation" in depths_in_use and "conf_map"  in depths_in_use:
                radar_depth= None; response = None
                image, radar_dilation, response, ground_truth, lidar_map = (
                    batch_data["image"], batch_data["radar_dilation"], batch_data["response"], batch_data["ground_truth"],batch_data["lidar_map"]
                )

                depth_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                depth_valid = depth_valid.bool()

                radar_dilation[~depth_valid] = 0.0  # set invalid dep

                # Apply transformations
                [image], [radar_dilation, response, ground_truth, lidar_map] = train_transforms.transform(
                    images_arr=[image],
                    range_maps_arr=[radar_dilation, response, ground_truth, lidar_map],
                    random_transform_probability=augmentation_probability
                )

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_dilation"] = radar_dilation
                batch_data["response"] = response
                batch_data["ground_truth"] = ground_truth
                batch_data["lidar_map"] = lidar_map

                # Combine depth and response if needed
                inputs = batch_data

            # Forward through the network
            output_depth,res_map = fusionnet_model.forward(
                inputs=inputs)

            # Compute loss function
            if ground_truth_dilation is not None:
                ground_truth = ground_truth_dilation(ground_truth)

            if ground_truth_outlier_removal is not None:
                ground_truth = ground_truth_outlier_removal.remove_outliers(ground_truth)   #删除lidar gt中的可能异常点
                
            validity_map_loss_smoothness = torch.where(
                ground_truth > 0,
                torch.zeros_like(ground_truth),
                torch.ones_like(ground_truth))

            loss, loss_info = fusionnet_model.compute_loss(
                image=image,
                output_depth=output_depth,
                ground_truth=ground_truth,
                lidar_map=lidar_map,
                loss_func=loss_func,
                w_smoothness=w_smoothness,
                loss_smoothness_kernel_size=loss_smoothness_kernel_size,
                validity_map_loss_smoothness=validity_map_loss_smoothness,
                w_lidar_loss=w_lidar_loss,
                w_dense_loss=w_dense_loss,
                w_perceptual_loss=w_perceptual_loss)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            if (train_step % 100) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / (train_step - train_step_retored) if radar_camera_fusionnet_restore_path is not None else (n_train_step - train_step) * time_elapse / train_step
                current_lr = optimizer.param_groups[0]['lr']
                train_info = 'Epoch={:3} Step={:6}/{} Lr={:.6f} Loss={:.5f}  Time Elapsed={:.2f}h Time Remaining={:.2f}h'.\
                format(epoch, train_step, n_train_step, current_lr, loss.item(), time_elapse, time_remain)
                pbar.update(batch_size*100)
                pbar.set_description(train_info)
                

            if (train_step % n_step_per_summary) == 0:
                mae,rmse,_,_ = get_metric(output_depth,lidar_map,min_evaluate_depth,max_evaluate_depth)
                
                log('Epoch={:3} Step={:6}/{} Lr={:.6f} Loss={:.5f} MAE={:10.4f} RMSE={:10.4f} Time Elapsed={:.2f}h Time Remaining={:.2f}h'.format(
                    epoch, train_step, n_train_step, current_lr, loss.item(), mae, rmse, time_elapse, time_remain),
                    log_path)

                with torch.no_grad():
                    # Log tensorboard summary
                    fusionnet_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        image=image,
                        input_depth=radar_depth,
                        radar_dilation=radar_dilation,
                        res_map = res_map,
                        input_response=response,
                        output_depth=output_depth,
                        ground_truth=ground_truth,
                        scalars=loss_info,
                        n_display=min(batch_size, 4))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint)== 0 and (train_step >= start_validation_step):

                log('Step={:6}/{} Lr={:.6f} Loss={:.5f} Time Elapsed={:.2f}h Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, current_lr, loss.item(), time_elapse, time_remain),
                    log_path)

                if (train_step % n_step_per_validation) == 0:
                    # Switch to validation mode
                    fusionnet_model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=fusionnet_model,
                            depths_in_use=depths_in_use,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            input_channels_depth = input_channels_depth,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            min_predict_depth=min_predict_depth,
                            max_predict_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_summary_display=4,
                            log_path=log_path)

                    # Switch back to training
                    fusionnet_model.train()

                # Save checkpoints
                fusionnet_model.save_model(
                    depth_model_checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=optimizer)

        pbar.close()

    # Evaluate once more after we are done training
    fusionnet_model.eval()

    with torch.no_grad():
        best_results = validate(
            model=fusionnet_model,
            depths_in_use=depths_in_use,
            dataloader=val_dataloader,
            transforms=val_transforms,
            input_channels_depth = input_channels_depth,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary_writer,
            n_summary_display=4,
            log_path=log_path)

    # Save checkpoints
    fusionnet_model.save_model(
        depth_model_checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)

def validate(model,
             depths_in_use,
             dataloader,
             transforms,
             input_channels_depth,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             max_predict_depth,
             min_predict_depth,
             device,
             summary_writer,
             n_summary_display=4,
             n_summary_display_interval=1000,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    input_depth_summary = []
    radar_dilation_summary = []
    response_summary = []
    ground_truth_summary = []
    res_map_summary = []

    for idx, batch_data in enumerate(dataloader):

        # Move inputs to device
        for key, in_ in batch_data.items():
            batch_data[key] = in_.to(device)

        if "radar" in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" in depths_in_use:
            # Unpack data
            image, radar_depth, radar_dilation, response, ground_truth= (
                batch_data["image"], batch_data["radar_depth"],
                batch_data["radar_dilation"], batch_data["response"],
                batch_data["ground_truth"]
            )

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)
            
            radar_dilation = apply_thr(quasi_depth=radar_dilation,conf_map=response,thr=0.4)
            
            # Validate and preprocess depth
            radar_depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
            radar_depth_valid = radar_depth_valid.bool()

            radar_dilation_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
            radar_dilation_valid = radar_dilation_valid.bool()

            radar_depth[~radar_depth_valid] = 0.0  # set invalid depth
            radar_dilation[~radar_dilation_valid] = 0.0

            # Update batch_data with transformed data
            batch_data["image"] = image
            batch_data["radar_depth"] = radar_depth
            batch_data["radar_dilation"] = radar_dilation
            batch_data["response"] = response
            batch_data["ground_truth"] = ground_truth

            # Combine depth and response if needed
            inputs = batch_data

        elif "radar" in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" not in depths_in_use:
            response = None
            image, radar_depth, radar_dilation,  ground_truth = (
                batch_data["image"], batch_data["radar_depth"],
                batch_data["radar_dilation"], 
                batch_data["ground_truth"]
            )

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Validate and preprocess depth
            radar_depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
            radar_depth_valid = radar_depth_valid.bool()

            radar_dilation_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
            radar_dilation_valid = radar_dilation_valid.bool()

            radar_depth[~radar_depth_valid] = 0.0  # set invalid depth
            radar_dilation[~radar_dilation_valid] = 0.0

            # Update batch_data with transformed data
            batch_data["image"] = image
            batch_data["radar_depth"] = radar_depth
            batch_data["radar_dilation"] = radar_dilation
            batch_data["ground_truth"] = ground_truth

            # Combine depth and response if needed
            inputs = batch_data

        elif "radar" in depths_in_use and "radar_dilation" not in depths_in_use and "conf_map" not in depths_in_use:
            response = None; radar_dilation=None
            image, radar_depth,  ground_truth = (
                batch_data["image"], batch_data["radar_depth"], batch_data["ground_truth"]
            )

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
            depth_valid = depth_valid.bool()
            
            radar_depth[~depth_valid] = 0.0  # set invalid dep

            # Update batch_data with transformed data
            batch_data["image"] = image
            batch_data["radar_depth"] = radar_depth
            batch_data["ground_truth"] = ground_truth

            # Combine depth and response if needed
            inputs = batch_data

        elif "radar" not in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" not in depths_in_use:
            radar_depth= None; response = None
            image, radar_dilation,  ground_truth = (
                batch_data["image"], batch_data["radar_dilation"], batch_data["ground_truth"]
            )

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            depth_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
            depth_valid = depth_valid.bool()

            radar_dilation[~depth_valid] = 0.0  # set invalid dep

            # Update batch_data with transformed data
            batch_data["image"] = image
            batch_data["radar_dilation"] = radar_dilation
            batch_data["ground_truth"] = ground_truth

            # Combine depth and response if needed
            inputs = batch_data

        elif "radar" not in depths_in_use and "radar_dilation" in depths_in_use and "conf_map"  in depths_in_use:
            radar_depth= None; response = None
            image, radar_dilation, response, ground_truth = (
                batch_data["image"], batch_data["radar_dilation"], batch_data["response"], batch_data["ground_truth"]
            )

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            depth_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
            depth_valid = depth_valid.bool()

            radar_dilation[~depth_valid] = 0.0  # set invalid dep

            # Update batch_data with transformed data
            batch_data["image"] = image
            batch_data["radar_dilation"] = radar_dilation
            batch_data["response"] = response
            batch_data["ground_truth"] = ground_truth

            # Combine depth and response if needed
            inputs = batch_data

        # Forward through network
        inputs_1 = inputs
        output_depth_1,res_map_1 = model.forward(
            inputs=inputs_1)

        inputs_2 = inputs
        for key_2,input_2 in inputs_2.items():
            inputs_2[key_2] = torch.flip(input_2, [3])
            
        output_depth_2,res_map_2 = model.forward(
            inputs=inputs_2)
        output_depth_2 = torch.flip(output_depth_2, [3])
        res_map_2 = torch.flip(res_map_2,[3]) if res_map_1 is not None else None
        
        output_depth = 0.5 * (output_depth_1 + output_depth_2)
        res_map = 0.5 * (res_map_1 + res_map_2) if res_map_1 is not None else None

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            radar_dilation_summary.append(radar_dilation)
            input_depth_summary.append(radar_depth)
            response_summary.append(response)
            ground_truth_summary.append(ground_truth)
            res_map_summary.append(res_map)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        validity_map = np.where(ground_truth > 0, 1, 0)

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Log to tensorboard
    if summary_writer is not None:
        model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image=torch.cat(image_summary, dim=0),
            input_depth=torch.cat(input_depth_summary, dim=0) if radar_depth is not None else None,
            radar_dilation=torch.cat(radar_dilation_summary, dim=0) if radar_dilation is not None else None,
            input_response=torch.cat(response_summary, dim=0) if response is not None else None,
            output_depth=torch.cat(output_depth_summary, dim=0),
            ground_truth=torch.cat(ground_truth_summary, dim=0),
            res_map=torch.cat(res_map_summary,dim=0) if res_map is not None else None,
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse},
            n_display=n_summary_display)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        imae=imae,
        irmse=irmse,
        step=step,
        log_path=log_path)

    if (np.round(mae, 2) <= np.round(best_results['mae'], 2)) and (np.round(rmse, 2) <= np.round(best_results['rmse'], 2)):
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log_evaluation_results(
        title='Best results',
        mae=best_results['mae'],
        rmse=best_results['rmse'],
        imae=best_results['imae'],
        irmse=best_results['irmse'],
        step=best_results['step'],
        log_path=log_path)

    return best_results

def get_metric(output_depth_t,ground_truth_t,min_evaluate_depth,max_evaluate_depth):

    with torch.no_grad():

        b,c,h,w = output_depth_t.shape
        mae = np.zeros(b)
        rmse = np.zeros(b)
        imae = np.zeros(b)
        irmse = np.zeros(b)
        
        output_depth_t = output_depth_t.cpu().detach().numpy()
        ground_truth_t = ground_truth_t.cpu().detach().numpy()

        for idx in range(b):
            output_depth = np.squeeze(output_depth_t[idx])
            ground_truth = np.squeeze(ground_truth_t[idx])

            validity_map = np.where(ground_truth > 0, 1, 0)

            # Select valid regions to evaluate
            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            # Compute validation metrics
            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

        # Compute mean metrics
        mae   = np.mean(mae)
        rmse  = np.mean(rmse)
        imae  = np.mean(imae)
        irmse = np.mean(irmse)

    return mae,rmse,imae,irmse

def run(
        restore_path,
        depths_in_use,
        image_path,
        depth_path,
        radar_dilation_path,
        response_path,
        ground_truth_path,
        # Input settings
        input_channels_image,
        input_channels_depth,
        normalized_image_range,
        # Network settings
        img_encoder_type,
        dep_encoder_type,
        n_filters_encoder_image,
        n_filters_encoder_depth,
        fusion_type,
        guidance_layers,
        fusion_layers,
        guidance,
        decoder_type,
        output_type,
        n_filters_decoder,
        n_resolutions_decoder,
        dropout_prob,
        min_predict_depth,
        max_predict_depth,
        # Weight settings
        weight_initializer,
        activation_func,
        # Output settings
        output_dirpath,
        save_outputs,
        keep_input_filenames,
        verbose=True,
        # Evaluation settings
        min_evaluate_depth=0.0,
        max_evaluate_depth=100.0):

    # Set up device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output directory
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    log_path = os.path.join(output_dirpath, 'results.txt')

    '''
    Set up paths for evaluation
    '''
    image_paths = data_utils.read_paths(image_path)

    n_sample = len(image_paths)

    depth_available = depth_path is not None and os.path.exists(depth_path)
    if depth_available:
        depth_paths = data_utils.read_paths(depth_path)

    radar_dilation_available = radar_dilation_path is not None and os.path.exists(radar_dilation_path)
    if radar_dilation_available:
        radar_dilation_paths = data_utils.read_paths(radar_dilation_path)
    else:
        radar_dilation_paths = [None] * n_sample

    ground_truth_available = \
        ground_truth_path is not None and \
        os.path.exists(ground_truth_path)

    if ground_truth_available:
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = [None] * n_sample

    response_available = response_path is not None and os.path.exists(response_path)
    if response_available:
        response_paths = data_utils.read_paths(response_path)
    else:
        response_paths = [None] * n_sample


    for paths in [depth_paths, radar_dilation_paths, response_paths, ground_truth_paths]:
        assert paths is None or n_sample == len(paths)

    dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetInferenceDataset(
            image_paths=image_paths,
            depth_paths=depth_paths,
            mono_depth_paths = None,
            radar_dilation_paths=radar_dilation_paths,
            response_paths=response_paths,
            ground_truth_paths=ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    '''
    Set up output paths
    '''
    if save_outputs:
        output_depth_radar_dirpath = os.path.join(output_dirpath)

        output_dirpaths = [
            output_depth_radar_dirpath,
        ]

        for dirpath in output_dirpaths:
            os.makedirs(dirpath, exist_ok=True)

    '''
    Build network and restore
    '''
    # Build network
    fusionnet_model = FusionNetModel(
        depths_in_use=depths_in_use,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        img_encoder_type=img_encoder_type,
        dep_encoder_type=dep_encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        guidance = guidance,
        guidance_layers = guidance_layers,
        fusion_layers = fusion_layers,
        decoder_type=decoder_type,
        n_resolution_decoder=n_resolutions_decoder,
        n_filters_decoder=n_filters_decoder,
        deconv_type='up',
        output_type=output_type,
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        dropout_prob=dropout_prob,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    fusionnet_model.eval()
    fusionnet_model.to(device)
    #fusionnet_model.data_parallel()

    parameters_fusionnet_model = fusionnet_model.parameters()

    step, _ = fusionnet_model.restore_model(restore_path)

    '''
    Log settings
    '''
    log('Evaluation input paths:', log_path)
    input_paths = [
        image_path,
       
    ]

    if depth_available:
        input_paths.append(depth_path)

    if radar_dilation_available:
        input_paths.append(radar_dilation_path)

    if ground_truth_available:
        input_paths.append(ground_truth_path)
    
    if response_available:
        input_paths.append(response_path)
        

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        depths_in_use=depths_in_use,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        img_encoder_type=img_encoder_type,
        dep_encoder_type=dep_encoder_type,
        frozen_strategy=None,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        fusion_type=fusion_type,
        guidance=guidance,
        guidance_layers=guidance_layers,
        fusion_layers=fusion_layers,
        decoder_type=decoder_type,
        output_type=output_type,
        n_filters_decoder=n_filters_decoder,
        n_resolutions_decoder=n_resolutions_decoder,
        dropout_prob=dropout_prob,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_fusionnet_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=output_dirpath,
        radar_camera_fusionnet_restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    if ground_truth_available:
        # Define evaluation metrics
        mae = np.zeros(n_sample)
        rmse = np.zeros(n_sample)
        imae = np.zeros(n_sample)
        irmse = np.zeros(n_sample)

    output_depth_fusion_paths = []

    with torch.no_grad():

        for idx, batch_data in enumerate(dataloader):

            # Move inputs to device
            for key, in_ in batch_data.items():
                batch_data[key] = in_.to(device)

            if "radar" in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" in depths_in_use:
                # Unpack data
                image, radar_depth, radar_dilation, response, ground_truth= (
                    batch_data["image"], batch_data["radar_depth"],
                    batch_data["radar_dilation"], batch_data["response"],
                    batch_data["ground_truth"]
                )

                [image] = transforms.transform(
                    images_arr=[image],
                    random_transform_probability=0.0)
                
                radar_dilation = apply_thr(radar_dilation,response,0.5,0.1)
                
                # Validate and preprocess depth
                radar_depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
                radar_depth_valid = radar_depth_valid.bool()

                radar_dilation_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                radar_dilation_valid = radar_dilation_valid.bool()
                
                # radar_depth_valid = (radar_depth < max_evaluate_depth) * (radar_depth > min_evaluate_depth)
                # radar_depth_valid = radar_depth_valid.bool()

                # radar_dilation_valid = (radar_dilation < max_evaluate_depth) * (radar_dilation > min_evaluate_depth)
                # radar_dilation_valid = radar_dilation_valid.bool()

                radar_depth[~radar_depth_valid] = 0.0  # set invalid depth
                radar_dilation[~radar_dilation_valid] = 0.0

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_depth"] = radar_depth
                batch_data["radar_dilation"] = radar_dilation
                batch_data["response"] = response
                batch_data["ground_truth"] = ground_truth

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" not in depths_in_use:
                response = None
                image, radar_depth, radar_dilation,  ground_truth = (
                    batch_data["image"], batch_data["radar_depth"],
                    batch_data["radar_dilation"], 
                    batch_data["ground_truth"]
                )

                [image] = transforms.transform(
                    images_arr=[image],
                    random_transform_probability=0.0)

                # Validate and preprocess depth
                radar_depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
                radar_depth_valid = radar_depth_valid.bool()

                radar_dilation_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                radar_dilation_valid = radar_dilation_valid.bool()

                radar_depth[~radar_depth_valid] = 0.0  # set invalid depth
                radar_dilation[~radar_dilation_valid] = 0.0

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_depth"] = radar_depth
                batch_data["radar_dilation"] = radar_dilation
                batch_data["ground_truth"] = ground_truth

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" in depths_in_use and "radar_dilation" not in depths_in_use and "conf_map" not in depths_in_use:
                response = None; radar_dilation=None
                image, radar_depth,  ground_truth = (
                    batch_data["image"], batch_data["radar_depth"], batch_data["ground_truth"]
                )

                [image] = transforms.transform(
                    images_arr=[image],
                    random_transform_probability=0.0)

                depth_valid = (radar_depth < max_predict_depth) * (radar_depth > min_predict_depth)
                depth_valid = depth_valid.bool()
                
                radar_depth[~depth_valid] = 0.0  # set invalid dep

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_depth"] = radar_depth
                batch_data["ground_truth"] = ground_truth

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" not in depths_in_use and "radar_dilation" in depths_in_use and "conf_map" not in depths_in_use:
                radar_depth= None; response = None
                image, radar_dilation,  ground_truth = (
                    batch_data["image"], batch_data["radar_dilation"], batch_data["ground_truth"]
                )

                [image] = transforms.transform(
                    images_arr=[image],
                    random_transform_probability=0.0)

                depth_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                depth_valid = depth_valid.bool()

                radar_dilation[~depth_valid] = 0.0  # set invalid dep

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_dilation"] = radar_dilation
                batch_data["ground_truth"] = ground_truth

                # Combine depth and response if needed
                inputs = batch_data

            elif "radar" not in depths_in_use and "radar_dilation" in depths_in_use and "conf_map"  in depths_in_use:
                radar_depth= None; response = None
                image, radar_dilation, response, ground_truth = (
                    batch_data["image"], batch_data["radar_dilation"], batch_data["response"], batch_data["ground_truth"]
                )

                [image] = transforms.transform(
                    images_arr=[image],
                    random_transform_probability=0.0)

                depth_valid = (radar_dilation < max_predict_depth) * (radar_dilation > min_predict_depth)
                depth_valid = depth_valid.bool()

                radar_dilation[~depth_valid] = 0.0  # set invalid dep

                # Update batch_data with transformed data
                batch_data["image"] = image
                batch_data["radar_dilation"] = radar_dilation
                batch_data["response"] = response
                batch_data["ground_truth"] = ground_truth

                # Combine depth and response if needed
                inputs = batch_data

            # Forward through network
            inputs_1 = inputs
            output_depth_1,res_map_1 = fusionnet_model.forward(
                inputs=inputs_1)
    
            inputs_2 = inputs
            for key_2,input_2 in inputs_2.items():
                inputs_2[key_2] = torch.flip(input_2, [3])
               
            output_depth_2,res_map_2 = fusionnet_model.forward(
                inputs=inputs_2)
            output_depth_2 = torch.flip(output_depth_2, [3])
            res_map_2 = torch.flip(res_map_2,[3]) if res_map_2 != None else None
            
            output_depth = 0.5 * (output_depth_1 + output_depth_2)
            res_map = 0.5 * (res_map_1 + res_map_2) if res_map_2 != None else None

            # output_depth,res_map = fusionnet_model.forward(
            #     inputs=inputs)

            output_depth_fusion = np.squeeze(output_depth.cpu().numpy())

            if verbose:
                print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

            '''
            Evaluate results
            '''
            if ground_truth_available:
                # Convert to numpy to validate
                ground_truth = np.squeeze(ground_truth.cpu().numpy())

                validity_map = np.where(ground_truth > 0, 1, 0)

                # Select valid regions to evaluate
                validity_mask = np.where(validity_map > 0, 1, 0)
                min_max_mask = np.logical_and(
                    ground_truth > min_evaluate_depth,
                    ground_truth < max_evaluate_depth)
                mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

                # Compute validation metrics
                mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth_fusion[mask], 1000.0 * ground_truth[mask])
                rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_fusion[mask], 1000.0 * ground_truth[mask])
                imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_fusion[mask], 0.001 * ground_truth[mask])
                irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_fusion[mask], 0.001 * ground_truth[mask])

            '''
            Save outputs
            '''
            if save_outputs:

                if keep_input_filenames:
                    filename = os.path.splitext(os.path.basename(ground_truth_paths[idx]))[0] + '.png'
                    scene = ground_truth_paths[idx].split("/")[6]
                    camera = ground_truth_paths[idx].split("/")[7]
                    output_depth_radar_dirpath_s = os.path.join(output_depth_radar_dirpath,scene,camera)


                # Create output paths
                os.makedirs(output_depth_radar_dirpath_s, exist_ok=True)
                output_depth_radar_path = os.path.join(output_depth_radar_dirpath_s, filename)

                #colorize
                output_depth_fusion = colorize(output_depth_fusion,min_evaluate_depth,max_evaluate_depth,cmap='viridis')

                Image.fromarray(output_depth_fusion).save(output_depth_radar_path)
                #data_utils.save_depth(output_depth_fusion, output_depth_radar_path)

        #         output_depth_fusion_paths.append(output_depth_radar_path)

        # if save_outputs:
        #    txt_name = os.path.basename(output_dirpath)
        #    data_utils.write_paths(image_path[:image_path.rfind('image')] + txt_name + ".txt",output_depth_fusion_paths)

    '''
    Print evaluation results
    '''
    if ground_truth_available:
        # Compute mean metrics
        mae   = np.mean(mae)
        rmse  = np.mean(rmse)
        imae  = np.mean(imae)
        irmse = np.mean(irmse)

        # Print evaluation results to console
        log_evaluation_results(
            title='Evaluation results',
            mae=mae,
            rmse=rmse,
            imae=imae,
            irmse=irmse,
            step=step,
            log_path=log_path)



'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       depths_in_use,
                       input_channels_image,
                       input_channels_depth,
                       normalized_image_range):

    log('Input settings:', log_path)
    log('depths_in_use={}'.format(depths_in_use),log_path)
    log('input_channels_image={}  input_channels_depth={}'.format(
        input_channels_image, input_channels_depth),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Network settings
                         img_encoder_type,
                         dep_encoder_type,
                         n_filters_encoder_image,
                         n_filters_encoder_depth,
                         fusion_type,
                         guidance,
                         guidance_layers,
                         fusion_layers,
                         decoder_type,
                         output_type,
                         frozen_strategy,
                         n_filters_decoder,
                         n_resolutions_decoder,
                         dropout_prob,
                         min_predict_depth,
                         max_predict_depth,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    log('Network settings:', log_path)
    log('img_encoder_type={}'.format(img_encoder_type),
        log_path)
    log('dep_encoder_type={}'.format(dep_encoder_type),
        log_path)
    log('fusion_type={}'.format(fusion_type),
        log_path)
    log('guidance={}'.format(guidance),
        log_path)
    log('guidance_layers={}'.format(guidance_layers),
        log_path)
    log('fusion_layers={}'.format(fusion_layers),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_filters_encoder_depth={}'.format(n_filters_encoder_depth),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('output_type={}'.format(output_type),
        log_path)
    log('frozen_strategy{}'.format(frozen_strategy),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
        log_path)
    log('n_resolutions_decoder={}'.format(
        n_resolutions_decoder),
        log_path)
    log('dropout_prob={}'.format(dropout_prob),
        log_path)
    log('min_predict_depth={}  max_predict_depth={}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('weight_initializer={}  activation_func={}'.format(
        weight_initializer, activation_func),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          batch_size,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_saturation,
                          augmentation_random_flip_type):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}  batch_size={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step, batch_size),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // batch_size), le * (n_train_sample // batch_size), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // batch_size), le * (n_train_sample // batch_size), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)

    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           loss_func,
                           w_smoothness,
                           w_weight_decay,
                           w_lidar_loss,
                           w_dense_loss,
                           w_perceptual_loss,
                           loss_smoothness_kernel_size,
                           outlier_removal_kernel_size,
                           outlier_removal_threshold,
                           ground_truth_dilation_kernel_size):

    log('Loss function settings:', log_path)
    log('loss_func={}'.format(
        loss_func),
        log_path)
    log('w_smoothness={:.1e}  w_weight_decay={:.1e}  w_lidar_loss={:.1e} w_perceptual_loss={:.1e} w_dense_loss={:.1e}'.format(
        w_smoothness, w_weight_decay, w_lidar_loss, w_perceptual_loss, w_dense_loss),
        log_path)
    log('loss_smoothness_kernel_size={}'.format(
        loss_smoothness_kernel_size),
        log_path)
    log('Ground truth preprocessing:')
    log('outlier_removal_kernel_size={}  outlier_removal_threshold={:.2f}'.format(
        outlier_removal_kernel_size, outlier_removal_threshold),
        log_path)
    log('dilation_kernel_size={}'.format(
        ground_truth_dilation_kernel_size),
        log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_dirpath=None,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_step_per_validation=None,
                        start_validation_step=None,
                        radar_camera_fusionnet_restore_path=None,
                        structralnet_restore_path=None,
                        transfer_type=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_dirpath is not None:
        log('checkpoint_path={}'.format(checkpoint_dirpath), log_path)

        if n_step_per_checkpoint is not None:
            log('n_step_per_checkpoint={}'.format(n_step_per_checkpoint), log_path)

        if n_step_per_validation is not None:
            log('n_step_per_validation={}'.format(n_step_per_validation), log_path)
            
        if start_validation_step is not None:
            log('start_validation_step={}'.format(start_validation_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_step_per_summary={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if  radar_camera_fusionnet_restore_path is not None and  radar_camera_fusionnet_restore_path != '':
        log(' radar_camera_fusionnet_restore_path={}'.format(radar_camera_fusionnet_restore_path),
            log_path)
        
    if  structralnet_restore_path is not None and  structralnet_restore_path != '':
        log(' structralnet_restore_path={}, transfer_type={}'.format(structralnet_restore_path,transfer_type),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)

def log_evaluation_results(title,
                           mae,
                           rmse,
                           imae,
                           irmse,
                           step=-1,
                           log_path=None):

    # Print evalulation results to console
    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step,
        mae,
        rmse,
        imae,
        irmse),
        log_path)
