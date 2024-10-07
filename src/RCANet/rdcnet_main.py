import sys
sys.path.append("/home/zfy/RCMDNet/src")
import os, time
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from utils import data_utils, eval_utils
from utils.log_utils import log
from RDCNet.rdcnet_model import RDCNetModel
from RDCNet.rdcnet_transforms import Transforms as rdcnet_transforms
from tqdm import tqdm
from datetime import datetime
import random
from datasets import datasets_rdcnet as datasets
from torchvision.utils import save_image

def train(
          train_image_path,
          train_radar_path,
          train_conf_ground_truth_path,
          val_image_path,
          val_radar_path,
          val_conf_ground_truth_path,
          # Batch settings
          batch_size,
          n_height,
          n_width,
          # Input settings
          input_channels_image,
          input_channels_radar_depth,
          normalized_image_range,
          # Network settings
          #rcnet
          rcnet_img_encoder_type,
          rcnet_dep_encoder_type,
          rcnet_n_filters_encoder_image,
          rcnet_n_filters_encoder_depth,
          rcnet_decoder_type,
          rcnet_n_filters_decoder,
          rcnet_fusion_type,
          rcnet_guidance,
          rcnet_guidance_layers,
          rcnet_fusion_layers,
          dropout_prob,
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
          #rcnet
          w_conf_loss,
          w_positive_class,
          set_invalid_to_negative_class,
          w_weight_decay,
          w_dense_loss,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          response_thr,
          # Checkpoint settings
          #checkpoint_dirpath,
          resultsave_dirpath,
          n_step_per_summary,
          n_step_per_checkpoint,
          n_step_per_validation,
          rdcnet_restore_path,
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

    save_dir_info = "_Input_" + "rd" if  "dilation" in train_radar_path else "_Input_" + "r"
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
        'ce': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    train_image_paths = data_utils.read_paths(train_image_path)
    train_radar_paths = data_utils.read_paths(train_radar_path)
    train_conf_ground_truth_paths = data_utils.read_paths(train_conf_ground_truth_path)

    n_train_sample = len(train_image_paths)

    for paths in [train_radar_paths,  train_conf_ground_truth_paths]:
        assert paths is None or n_train_sample == len(paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    # Set up data loader and data transforms
    train_dataloader = torch.utils.data.DataLoader(
        datasets.RDCNetTrainingDataset(
            #"rcnet"
            image_paths=train_image_paths,
            radar_paths=train_radar_paths,
            conf_ground_truth_paths=train_conf_ground_truth_paths,
            shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_thread)
    

    train_transforms = rdcnet_transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation,
        random_flip_type=augmentation_random_flip_type)

    '''
    Set up paths for validation
    '''
    val_image_paths = data_utils.read_paths(val_image_path)
    val_radar_paths = data_utils.read_paths(val_radar_path)
    val_conf_ground_truth_paths = data_utils.read_paths(val_conf_ground_truth_path)

    n_val_sample = len(val_image_paths)

    for paths in [val_radar_paths,  val_conf_ground_truth_paths]:
        assert paths is None or n_val_sample == len(paths)

    val_dataloader = torch.utils.data.DataLoader(
        datasets.RDCNetInferenceDataset(
            #fusionnet
            image_paths=val_image_paths,
            radar_paths=val_radar_paths,
            conf_ground_truth_paths=val_conf_ground_truth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)
    
    val_transforms_rdcnet = rdcnet_transforms(
        normalized_image_range=normalized_image_range)
    
    points_dilation = torch.nn.MaxPool2d(
        kernel_size=5,
        stride=1,
        padding=5 // 2)

    '''
    Set up the model
    '''
    # FusionNet
    rdcnet_model = RDCNetModel(
        #rcnet
        input_channels_image=input_channels_image,
        input_channels_radar_depth=input_channels_radar_depth,
        rcnet_n_filters_encoder_image=rcnet_n_filters_encoder_image,
        rcnet_n_filters_encoder_depth=rcnet_n_filters_encoder_depth,
        rcnet_img_encoder_type=rcnet_img_encoder_type,
        rcnet_dep_encoder_type=rcnet_dep_encoder_type,
        rcnet_decoder_type=rcnet_decoder_type,
        rcnet_n_filters_decoder=rcnet_n_filters_decoder,
        rcnet_fusion_type=rcnet_fusion_type,
        rcnet_guidance=rcnet_guidance,
        rcnet_guidance_layers=rcnet_guidance_layers,
        rcnet_fusion_layers=rcnet_fusion_layers,
        #public
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        dropout_prob=dropout_prob,
        max_predict_depth=max_evaluate_depth,
        min_predict_depth=min_evaluate_depth,
        device=device)

    rdcnet_model.to(device)
    #fusionnet_model.data_parallel()

    if len(cuda_visible_devices) >1:
        rdcnet_model.data_parallel()

    parameters_rdcnet_model = rdcnet_model.parameters()

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_image_path,
        train_radar_path
    ]
    for path in train_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_radar_path
    ]

    for path in val_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        input_channels_image=input_channels_image,
        input_channels_quasi_depth=input_channels_radar_depth,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        #rcnet
        rcnet_img_encoder_type=rcnet_img_encoder_type,
        rcnet_dep_encoder_type=rcnet_dep_encoder_type,
        rcnet_n_filters_encoder_image=rcnet_n_filters_encoder_image,
        rcnet_n_filters_encoder_depth=rcnet_n_filters_encoder_depth,
        rcnet_decoder_type=rcnet_decoder_type,
        rcnet_n_filters_decoder=rcnet_n_filters_decoder,
        rcnet_fusion_type=rcnet_fusion_type,
        rcnet_guidance= rcnet_guidance,
        rcnet_guidance_layers=rcnet_guidance_layers,
        rcnet_fusion_layers=rcnet_fusion_layers,
        dropout_prob=dropout_prob,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_rdcnet_model)

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
        #rcnet
        w_conf_loss=w_conf_loss,
        w_positive_class=w_positive_class,
        set_invalid_to_negative_class=set_invalid_to_negative_class,
        w_weight_decay=w_weight_decay,
        w_dense_loss=w_dense_loss)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth,
        response_thr=response_thr)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=resultsave_dirpath,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        n_step_per_validation=n_step_per_validation,
        restore_path=rdcnet_restore_path,
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
            'params': parameters_rdcnet_model,
            'weight_decay': w_weight_decay
        }],
        lr=learning_rates[0]  # 初始学习率
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rates[0],  # 使用最大学习率
        total_steps=int(n_train_step)+10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear',
        last_epoch=-1,
        three_phase=False,
    )

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    # Start training
    train_step = 0

    if rdcnet_restore_path is not None and rdcnet_restore_path != '':
        train_step_retored, optimizer = rdcnet_model.restore_model(
            rdcnet_restore_path,
            optimizer=optimizer)
        ################################################################
        # n_train_step += train_step_retored
        ################################################################
        train_step = train_step_retored

        lr_scheduler.last_epoch = train_step

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
            rcnet_inputs = batch_data

            # Fetch data
            for key, in_ in rcnet_inputs.items():
                rcnet_inputs[key] = in_.to(device)

            # Apply augmentations and data transforms
            image, radar_depth, conf_ground_truth = (
                rcnet_inputs["image"], rcnet_inputs["radar_depth"], 
                rcnet_inputs["conf_ground_truth"]
            )
            
            
            # Apply transformations
            [image], [radar_depth, conf_ground_truth] = train_transforms.transform(
                images_arr=[image],
                range_maps_arr=[radar_depth, conf_ground_truth],
                random_transform_probability=augmentation_probability
            )

            # Update batch_data with transformed data
            rcnet_inputs["image"]               = image
            rcnet_inputs["radar_depth"]        = radar_depth

            # Forward through the network
            conf_map_logits,conf_map = rdcnet_model.forward(
                rcnet_inputs=rcnet_inputs)
            
            quasi_depth = apply_thr(radar_depth,conf_map,response_thr)
            valid_depth = (conf_ground_truth > 0.0)
            ground_truth = radar_depth*valid_depth
            
            # -------------------GT for rcnet--------------------------------------------
            # Mask out invalid pixels in loss
            validity_map = torch.where(
                radar_depth <= 0,
                torch.zeros_like(radar_depth),
                torch.ones_like(radar_depth))
          # --------------------------------------------------------------------------------

            loss, loss_info = rdcnet_model.compute_loss(
                #rcnet
                logits=conf_map_logits,
                ground_truth_label=conf_ground_truth,
                validity_map=validity_map,
                w_positive_class=w_positive_class,
                w_conf_loss=w_conf_loss,
                w_dense_loss=w_dense_loss,
                )

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            if (train_step % 100) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / (train_step - train_step_retored) if rdcnet_restore_path is not None else (n_train_step - train_step) * time_elapse / train_step
                current_lr = optimizer.param_groups[0]['lr']
                train_info = 'Epoch={:3} Step={:6}/{} Lr={:.6f} Loss={:.5f}  Time Elapsed={:.2f}h Time Remaining={:.2f}h'.\
                format(epoch, train_step, n_train_step, current_lr, loss.item(), time_elapse, time_remain)
                pbar.update(batch_size*100)
                pbar.set_description(train_info)

            if (train_step % n_step_per_summary) == 0:

                log('Epoch={:3} Step={:6}/{} Loss={:.5f} Time Elapsed={:.2f}h Time Remaining={:.2f}h'.format(
                    epoch, train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                with torch.no_grad():
                    # Log tensorboard summary
                    rdcnet_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        image=image,
                        radar_depth=radar_depth,
                        input_response=conf_map,
                        input_depth=quasi_depth,
                        ground_truth = ground_truth,
                        ground_truth_label=conf_ground_truth,
                        scalars=loss_info,
                        n_display=min(batch_size, 4))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if (train_step % n_step_per_validation) == 0:
                    # Switch to validation mode
                    rdcnet_model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=rdcnet_model,
                            dataloader=val_dataloader,
                            rdcnet_transforms=val_transforms_rdcnet,
                            step=train_step,
                            w_positive_class=w_positive_class,
                            w_conf_loss=w_conf_loss,
                            w_dense_loss=w_dense_loss,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_summary_display=4,
                            log_path=log_path)

                    # Switch back to training
                    rdcnet_model.train()

                # Save checkpoints
                rdcnet_model.save_model(
                    depth_model_checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=optimizer)
        pbar.close()

    # Evaluate once more after we are done training
    rdcnet_model.eval()

    with torch.no_grad():
        best_results = validate(
                model=rdcnet_model,
                dataloader=val_dataloader,
                rdcnet_transforms=val_transforms_rdcnet,
                step=train_step,
                w_positive_class=w_positive_class,
                w_conf_loss=w_conf_loss,
                w_dense_loss=w_dense_loss,
                best_results=best_results,
                device=device,
                summary_writer=val_summary_writer,
                min_evaluate_depth=min_evaluate_depth,
                max_evaluate_depth=max_evaluate_depth,
                n_summary_display=4,
                log_path=log_path)

    # Save checkpoints
    rdcnet_model.save_model(
        depth_model_checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)

    with torch.no_grad():
        rdcnet_model.log_summary(
            summary_writer=train_summary_writer,
            tag='train',
            step=train_step,
            image=image,
            radar_depth=radar_depth,
            input_response=conf_map,
            input_depth=quasi_depth,
            ground_truth = ground_truth,
            ground_truth_label=conf_ground_truth,
            scalars=loss_info,
            n_display=min(batch_size, 4))

    # Save checkpoints
    rdcnet_model.save_model(
        depth_model_checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)

def validate(model,
             dataloader,
             rdcnet_transforms,
             step,
             w_positive_class,
             w_conf_loss,
             w_dense_loss,
             best_results,
             device,
             summary_writer,
             n_summary_display=4,
             n_summary_display_interval=1000,
             min_evaluate_depth=0.0,
             max_evaluate_depth=100.0,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    ce = np.zeros(n_sample)

    image_summary = []
    response_summary = []
    radar_depth_summary= []
    input_depth_summary = []
    conf_ground_truth_summary = []
    ground_truth_summary = []

    for idx, batch_data in enumerate(dataloader):

        rcnet_inputs = batch_data

        # Move inputs to device
        for key, in_ in rcnet_inputs.items():
            rcnet_inputs[key] = in_.to(device)

        image, radar_depth, conf_ground_truth = (
                rcnet_inputs["image"], rcnet_inputs["radar_depth"], 
                rcnet_inputs["conf_ground_truth"]
            )


        [image] = rdcnet_transforms.transform(
            images_arr=[image],
            random_transform_probability=0.0)
        
        # Update batch_data with transformed data
        rcnet_inputs["image"]               = image
        rcnet_inputs["radar_depth"]         = radar_depth
       
        # Forward through network
        conf_map_logits,conf_map = model.forward(
                rcnet_inputs=rcnet_inputs)
        
        quasi_depth = apply_thr(radar_depth,conf_map,0.5)
        valid_depth = (conf_ground_truth > 0.0).float()
        ground_truth = radar_depth*valid_depth

        validity_map = torch.where(
                radar_depth <= 0,
                torch.zeros_like(radar_depth),
                torch.ones_like(radar_depth))

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            image_summary.append(image),
            radar_depth_summary.append(radar_depth),
            input_depth_summary.append(quasi_depth),
            response_summary.append(conf_map),
            conf_ground_truth_summary.append(conf_ground_truth),
            ground_truth_summary.append(ground_truth)

        ce_i,loss_info = model.compute_loss(
                #rcnet
                logits=conf_map_logits,
                ground_truth_label=conf_ground_truth,
                validity_map=validity_map,
                w_positive_class=w_positive_class,
                w_conf_loss=w_conf_loss,
                w_dense_loss=w_dense_loss,
                )

        # Convert to numpy to validate
        quasi_depth = np.squeeze(quasi_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())
        ce_i = ce_i.cpu().numpy()


        # Select valid regions to evaluate
        validity_mask = np.where(np.squeeze(radar_depth.cpu().numpy()) > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        quasi_depth = quasi_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * quasi_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * quasi_depth, 1000.0 * ground_truth)
        ce[idx] = ce_i

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    ce    = np.mean(ce)

    # Log to tensorboard
    if summary_writer is not None:
        model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image=torch.cat(image_summary, dim=0),
            radar_depth=torch.cat(radar_depth_summary,dim=0),
            input_depth=torch.cat(input_depth_summary, dim=0),
            input_response=torch.cat(response_summary, dim=0) if conf_map is not None else None,
            ground_truth=torch.cat(ground_truth_summary, dim=0),
            ground_truth_label=torch.cat(conf_ground_truth_summary) if conf_ground_truth is not  None else None,
            scalars={'mae' : mae, 'rmse' : rmse, 'ce' : ce},
            n_display=n_summary_display)

    # Print validation results to console
    log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        ce=ce,
        step=step,
        log_path=log_path)

    if (np.round(ce, 2) <= np.round(best_results['ce'], 2)):
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['ce'] = ce

    log_evaluation_results(
        title='Best results',
        mae=best_results['mae'],
        rmse=best_results['rmse'],
        ce=best_results['ce'],
        step=best_results['step'],
        log_path=log_path)

    return best_results

def run(
        restore_path,
        image_path,
        radar_path,
        conf_ground_truth_path,
        # Input settings
        input_channels_image,
        input_channels_radar_depth,
        normalized_image_range,
        # Network settings
        #rcnet
        rcnet_img_encoder_type,
        rcnet_dep_encoder_type,
        rcnet_n_filters_encoder_image,
        rcnet_n_filters_encoder_depth,
        rcnet_decoder_type,
        rcnet_n_filters_decoder,
        rcnet_fusion_type,
        rcnet_guidance,
        rcnet_guidance_layers,
        rcnet_fusion_layers,
        dropout_prob,
        # Weight settings
        weight_initializer,
        activation_func,
        w_positive_class,
        # Files
        output_dirpath,
        save_outputs,
        keep_input_filenames,
        verbose=True,
        # Evaluation settings
        min_evaluate_depth=0.0,
        max_evaluate_depth=100.0,
        response_thr=0.5):

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
    radar_paths = data_utils.read_paths(radar_path)

    n_sample = len(image_paths)

    conf_ground_truth_paths = data_utils.read_paths(conf_ground_truth_path)

    for paths in [radar_paths, conf_ground_truth_paths]:
        assert paths is None or n_sample == len(paths)

    dataloader = torch.utils.data.DataLoader(
        datasets.RDCNetInferenceDataset(
            #fusionnet
            image_paths=image_paths,
            radar_paths=radar_paths,
            conf_ground_truth_paths=conf_ground_truth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)
    
    transforms = rdcnet_transforms(
        normalized_image_range=normalized_image_range)

    '''
    Set up output paths
    '''
    if save_outputs:
        output_depth_dirpath = os.path.join(output_dirpath, 'selected_dilation_radar')
        output_response_radar_dirpath = os.path.join(output_dirpath, 'conf_map')
        output_ground_truth_dirpath = os.path.join(output_dirpath, 'gt_selected_dilation_radar')

        output_dirpaths = [
            output_depth_dirpath,
            output_response_radar_dirpath,
            output_ground_truth_dirpath
        ]

        for dirpath in output_dirpaths:
            os.makedirs(dirpath, exist_ok=True)

    '''
    Build network and restore
    '''
    # Build network
    rdcnet_model = RDCNetModel(
        #rcnet
        input_channels_image=input_channels_image,
        input_channels_radar_depth=input_channels_radar_depth,
        rcnet_n_filters_encoder_image=rcnet_n_filters_encoder_image,
        rcnet_n_filters_encoder_depth=rcnet_n_filters_encoder_depth,
        rcnet_img_encoder_type=rcnet_img_encoder_type,
        rcnet_dep_encoder_type=rcnet_dep_encoder_type,
        rcnet_decoder_type=rcnet_decoder_type,
        rcnet_n_filters_decoder=rcnet_n_filters_decoder,
        rcnet_fusion_type=rcnet_fusion_type,
        rcnet_guidance=rcnet_guidance,
        rcnet_guidance_layers=rcnet_guidance_layers,
        rcnet_fusion_layers=rcnet_fusion_layers,
        #public
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        dropout_prob=dropout_prob,
        max_predict_depth=max_evaluate_depth,
        min_predict_depth=min_evaluate_depth,
        device=device)
  
    rdcnet_model.eval()
    rdcnet_model.to(device)
    #fusionnet_model.data_parallel()

    parameters_rdcnet_model = rdcnet_model.parameters()
    step, _ = rdcnet_model.restore_model(restore_path)

    '''
    Log settings
    '''
    log('Evaluation input paths:', log_path)
    input_paths = [
        image_path,
        radar_path
    ]

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        input_channels_image=input_channels_image,
        input_channels_quasi_depth=input_channels_radar_depth,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Network settings
        #rcnet
        rcnet_img_encoder_type=rcnet_img_encoder_type,
        rcnet_dep_encoder_type=rcnet_dep_encoder_type,
        rcnet_n_filters_encoder_image=rcnet_n_filters_encoder_image,
        rcnet_n_filters_encoder_depth=rcnet_n_filters_encoder_depth,
        rcnet_decoder_type=rcnet_decoder_type,
        rcnet_n_filters_decoder=rcnet_n_filters_decoder,
        rcnet_fusion_type=rcnet_fusion_type,
        rcnet_guidance= rcnet_guidance,
        rcnet_guidance_layers=rcnet_guidance_layers,
        rcnet_fusion_layers=rcnet_fusion_layers,
        dropout_prob=dropout_prob,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_rdcnet_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth,
        response_thr=response_thr)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_dirpath=output_dirpath,
        restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    ce = np.zeros(n_sample)

    output_depth_paths = []
    output_response_radar_paths = []
    output_ground_truth_paths = []

    with torch.no_grad():

        for idx, batch_data in enumerate(dataloader):

            rcnet_inputs = batch_data

            # Move inputs to device
            for key, in_ in rcnet_inputs.items():
                rcnet_inputs[key] = in_.to(device)

            image, radar_depth, conf_ground_truth = (
                    rcnet_inputs["image"], rcnet_inputs["radar_depth"], 
                    rcnet_inputs["conf_ground_truth"]
                )


            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)
            
            # Update batch_data with transformed data
            rcnet_inputs["image"]               = image
            rcnet_inputs["radar_depth"]         = radar_depth
        
            # Forward through network
            
            conf_map_logits,conf_map = rdcnet_model.forward(
                    rcnet_inputs=rcnet_inputs)
            
            quasi_depth = apply_thr(radar_depth,conf_map,response_thr)
            valid_depth = (conf_ground_truth > 0.0).float()
            ground_truth = radar_depth*valid_depth

            ce_validity_map = torch.where(
                    radar_depth <= 0,
                    torch.zeros_like(radar_depth),
                    torch.ones_like(radar_depth))


            output_depth = np.squeeze(quasi_depth.cpu().numpy())
            output_conf_map = np.squeeze(conf_map.cpu().numpy())
            output_ground_truth = np.squeeze(ground_truth.cpu().numpy())

            if verbose:
                print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

            '''
            Evaluate results
            '''
            ce_i,loss_info = rdcnet_model.compute_loss(
                #rcnet
                logits=conf_map_logits,
                ground_truth_label=conf_ground_truth,
                validity_map=ce_validity_map,
                w_positive_class=w_positive_class,
                w_conf_loss=0.0,
                w_dense_loss=0.0,
                )
            
            # Convert to numpy to validate
            quasi_depth = np.squeeze(quasi_depth.cpu().numpy())
            ground_truth = np.squeeze(ground_truth.cpu().numpy())
            ce_i = ce_i.cpu().numpy()

            validity_map = np.where(ground_truth > 0, 1, 0)

            # Select valid regions to evaluate
            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            quasi_depth = quasi_depth[mask]
            ground_truth = ground_truth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * quasi_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * quasi_depth, 1000.0 * ground_truth)
            ce[idx] = ce_i

            '''
            Save outputs
            '''
            if save_outputs:

                if keep_input_filenames:
                    filename = os.path.splitext(os.path.basename(radar_paths[idx]))[0] + '.png'
                    scene = radar_paths[idx].split("/")[6]
                    camera = radar_paths[idx].split("/")[7]
                    output_depth_dirpath_s = os.path.join(output_depth_dirpath, scene, camera)
                    output_response_radar_dirpath_s = os.path.join(output_response_radar_dirpath, scene, camera)
                    #output_ground_truth_dirpath_s = os.path.join(output_ground_truth_dirpath, scene, camera)
                else:
                    filename = '{:010d}.png'.format(idx)
                    scene = radar_paths[idx].split("/")[6]
                    camera = radar_paths[idx].split("/")[7]
                    output_depth_dirpath_s = os.path.join(output_depth_dirpath, scene, camera)
                    output_response_radar_dirpath_s = os.path.join(output_response_radar_dirpath, scene, camera)
                    #output_ground_truth_dirpath_s = os.path.join(output_ground_truth_dirpath, scene, camera)

                os.makedirs(output_depth_dirpath_s, exist_ok=True)
                os.makedirs(output_response_radar_dirpath_s, exist_ok=True)
                #os.makedirs(output_ground_truth_dirpath_s, exist_ok=True)

                # Create output paths
                output_depth_path = os.path.join(output_depth_dirpath_s, filename)
                output_response_radar_path = os.path.join(output_response_radar_dirpath_s, filename) 
                #output_ground_truth_path = os.path.join(output_ground_truth_dirpath_s, filename) 

                data_utils.save_depth(output_depth, output_depth_path)
                data_utils.save_depth(output_conf_map,output_response_radar_path)
                #data_utils.save_depth(output_ground_truth,output_ground_truth_path)

                output_depth_paths.append(output_depth_path)
                output_response_radar_paths.append(output_response_radar_path)
                #output_ground_truth_paths.append(output_ground_truth_path)

        # if save_outputs:
        #    tag = image_path.split("/")[0]
        #    txt_path = os.path.join(tag,"nuscenes")
        #    data_utils.write_paths(os.path.join(txt_path,"nuscenes_" + tag[:-3] + "_selected_dilation_radar_attwp.txt"),output_depth_paths)
        #    data_utils.write_paths(os.path.join(txt_path,"nuscenes_" + tag[:-3] + "_conf_.txt"),output_response_radar_paths)
        #    #data_utils.write_paths(os.path.join(txt_path,"nuscenes_" + tag[:-3] + "_gt_selected_dilation_radar_attwp.txt"),output_ground_truth_paths)



    '''
    Print evaluation results
    '''
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    ce    = np.mean(ce)

    log_evaluation_results(
        title='Validation results',
        mae=mae,
        rmse=rmse,
        ce=ce,
        step=step,
        log_path=log_path)


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       input_channels_image,
                       input_channels_quasi_depth,
                       normalized_image_range):

    log('Input settings:', log_path)
    log('input_channels_image={}  input_channels_quasi_depth={}'.format(
        input_channels_image, input_channels_quasi_depth),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Network settings
                         #rcnet
                         rcnet_img_encoder_type,
                         rcnet_dep_encoder_type,
                         rcnet_n_filters_encoder_image,
                         rcnet_n_filters_encoder_depth,
                         rcnet_decoder_type,
                         rcnet_n_filters_decoder,
                         rcnet_fusion_type,
                         rcnet_guidance,
                         rcnet_guidance_layers,
                         rcnet_fusion_layers,
                         dropout_prob,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    log('Network settings:', log_path)
    log('--------RCNet--------:',log_path)
    log('rcnet_img_encoder_type={}'.format(rcnet_img_encoder_type),
        log_path)
    log('rcnet_dep_encoder_type={}'.format(rcnet_dep_encoder_type),
        log_path)
    log('rcnet_n_filters_encoder_image={}'.format(rcnet_n_filters_encoder_image),
        log_path)
    log('rcnet_n_neurons_encoder_depth={}'.format(rcnet_n_filters_encoder_depth),
        log_path)
    log('rcnet_decoder_type={}'.format(rcnet_decoder_type),
        log_path)
    log('rcnet_n_filters_decoder={}'.format(rcnet_n_filters_decoder),
        log_path)
    log('rcnet_fusion_type={}'.format(rcnet_fusion_type),
        log_path)
    log('rcnet_guidance={}'.format(rcnet_guidance),
        log_path)
    log('rcnet_guidance_layers={}'.format(rcnet_guidance_layers),
        log_path)
    log('rcnet_fusion_layers={}'.format(rcnet_fusion_layers),
        log_path)
    log('dropout_prob={}'.format(dropout_prob),
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
                          augmentation_random_flip_type
                          ):

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
                           #rcnet
                           w_conf_loss,
                           w_positive_class,
                           set_invalid_to_negative_class,
                           #fusionnet
                           w_weight_decay,
                           w_dense_loss):

    log('Loss function settings:', log_path)
    log('--------rcnet----------:', log_path)
    log('w_conf_loss={}'.format(w_conf_loss),log_path)
    log('w_positive_class={}'.format(w_positive_class),log_path)
    log('set_invalid_to_negative_class={}'.format(set_invalid_to_negative_class),log_path)

    log('--------fusionnet----------:', log_path)
    log('w_weight_decay={:.1e}  w_dense_loss={:.1e}'.format(
        w_weight_decay, w_dense_loss),
        log_path)
    log('Ground truth preprocessing:')
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth,
                            response_thr):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('response_thr={:.2f}'.format(response_thr), log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_dirpath=None,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_step_per_validation=None,
                        restore_path=None,
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

    if  restore_path is not None and  restore_path != '':
        log(' restore_path={}'.format(restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)

def log_evaluation_results(title,
                           mae,
                           rmse,
                           ce,
                           step=-1,
                           log_path=None):

    # Print evalulation results to console
    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8} '.format(
        'Step', 'MAE', 'RMSE', 'CE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f} '.format(
        step,
        mae,
        rmse,
        ce),
        log_path)

def apply_thr(quasi_depth,conf_map,thr):
    #valid_mask = torch.sigmoid(1e3 * (conf_map - thr)).float()
    valid_mask = (conf_map>thr).float()
    quasi_depth = quasi_depth*valid_mask
    while(quasi_depth[quasi_depth>1e-3].shape == 0):
        thr=thr-0.05
        #valid_mask = torch.sigmoid(1e3 * (conf_map - thr)).float()
        valid_mask = (conf_map>thr).float()
        quasi_depth = quasi_depth*valid_mask
    return quasi_depth