import torch, torchvision
from utils import log_utils
from utils.data_utils import save_feature_maps,hist,save_feature_maps_batch
import rdcnet_losses as losses
import RDCNet.networks as rcmdnetworks
import numpy as np
from utils.net_utils import param_transfer
import torch.nn as nn

class RDCNetModel(nn.Module):
    '''
    Image radar fusion

    Arg(s):
        encoder_type : str
            encoder type
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        n_filters_encoder_image : list[int]
            list of filters for each layer in encoder
        n_filters_encoder_image : list[int]
            list of filters for each layer in encoder
        decoder_type : str
            decoder type
        n_resolution_decoder : int
            minimum resolution of multiscale outputs is 1/(2^n_resolution_decoder)
        n_filters_decoder : list[int]
            list of filters for each layer in decoder
        resolutions_depthwise_separable_decoder : list[int]
            resolutions to use depthwise separable convolution
        output_func: str
            output function of decoder
        activation_func : str
            activation function for network
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        use_batch_norm : bool
            if set, then applied batch normalization
        min_predict_depth : float
            minimum predicted depth
        max_predict_depth : float
            maximum predicted depth
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 #rcnet
                 input_channels_image,
                 input_channels_radar_depth,
                 rcnet_n_filters_encoder_image,
                 rcnet_n_filters_encoder_depth,
                 rcnet_img_encoder_type,
                 rcnet_dep_encoder_type,
                 rcnet_decoder_type,
                 rcnet_n_filters_decoder,
                 rcnet_fusion_type,
                 rcnet_guidance,
                 rcnet_guidance_layers,
                 rcnet_fusion_layers,
                 #public
                 activation_func,
                 weight_initializer,
                 dropout_prob,
                 max_predict_depth,
                 min_predict_depth,
                 device=torch.device('cuda')):
        
        super(RDCNetModel, self).__init__()
        #------------------------------Public---------------------------------
        self.device = device
        self.input_channels_image = input_channels_image
        self.max_predict_depth =max_predict_depth
        self.device = device

        # -----------------------Build RCNet encoder--------------------------
        if rcnet_fusion_type in ['attention_wp','add','weight', 'weight_and_project','SVFF','SVFF_add','cagf','attention']:
            rcnet_latent_channels = rcnet_n_filters_encoder_image[-1]

        elif rcnet_fusion_type == 'concat':
            n_filters_encoder = [
                i + z
                for i, z in zip(rcnet_n_filters_encoder_image, rcnet_n_filters_encoder_depth)
            ]
            rcnet_latent_channels = n_filters_encoder[-1]
        else:
            raise ValueError('Unsupported fusion type: {}'.format(rcnet_fusion_type))

        if 'resnet18' in rcnet_img_encoder_type:
            rcnet_img_n_layer = 18
        elif 'resnet34' in rcnet_img_encoder_type:
            rcnet_img_n_layer = 34
        else:
            raise ValueError('Unsupported encoder type: {}'.format(rcnet_img_encoder_type))
        
        if 'resnet18' in rcnet_dep_encoder_type:
            rcnet_dep_n_layer = 18
        elif 'resnet34' in rcnet_dep_encoder_type:
            rcnet_dep_n_layer = 34
        else:
            raise ValueError('Unsupported encoder type: {}'.format(rcnet_dep_encoder_type))

        self.rcnet_encoder = rcmdnetworks.RCNetEncoder(
            n_img_layer=rcnet_img_n_layer,
            n_dep_layer=rcnet_dep_n_layer,
            input_channels_image=input_channels_image,
            input_channels_depth=input_channels_radar_depth,
            n_filters_encoder_image=rcnet_n_filters_encoder_image,
            n_filters_encoder_depth=rcnet_n_filters_encoder_depth,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm='batch_norm' in rcnet_img_encoder_type,
            dropout_prob=dropout_prob,
            fusion_type=rcnet_fusion_type,
            guidance=rcnet_guidance,
            guidance_layers=rcnet_guidance_layers,
            fusion_layers=rcnet_fusion_layers,
            device=device)

        # Calculate number of channels for latent and skip connections combining image + depth
        n_skips_rcnet = rcnet_n_filters_encoder_image[:-1]
        n_skips_rcnet = n_skips_rcnet[::-1] + [0]

        # Build RCNet decoder
        if 'multiscale' in rcnet_decoder_type:
            self.rcnet_decoder = rcmdnetworks.RCNetDecoder(
                input_channels=rcnet_latent_channels,
                output_channels=1,
                n_resolution=1,
                n_filters=rcnet_n_filters_decoder,
                n_skips=n_skips_rcnet,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in rcnet_decoder_type,
                dropout_prob=dropout_prob,
                deconv_type='up'
                )
        else:
            raise ValueError('Decoder type {} not supported.'.format(rcnet_decoder_type))

        
    def forward(self, rcnet_inputs):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            input_depth : torch.Tensor[float32]
                N x 1 x H x W input depth
            return_multiscale : bool
                if set, then return multiple outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''
        
        #uzip data
        image, radar_depth= (rcnet_inputs["image"], rcnet_inputs["radar_depth"])

        #rcnet forward
        rc_latent, rc_skips, = self.rcnet_encoder(image, radar_depth)
        outputs,conf_map_logits = self.rcnet_decoder(x=rc_latent, skips=rc_skips,shape=image.shape[-2:])
        conf_map = torch.sigmoid(conf_map_logits)

        return conf_map_logits,conf_map

            
    def compute_loss(self,
                     #rcnet
                     logits,
                     ground_truth_label,
                     validity_map,
                     w_positive_class,
                     w_conf_loss,
                     w_dense_loss,
                     ):
        '''
        Computes loss function

        Arg(s):
            image :  torch.Tensor[float32]
                N x 3 x H x W image
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth
            lidar_map : torch.Tensor[float32]
                N x 1 x H x W single lidar scan
            loss_func : str
                loss function to minimize
            w_smoothness : float
                weight of local smoothness loss
            loss_smoothness_kernel_size : tuple[int]
                kernel size of loss smoothness
            validity_map_loss_smoothness : torch.Tensor[float32]
                N x 1 x H x W validity map
            w_lidar_loss : float
                weight of lidar loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

       
        #----------------------------rcnet-----------------------------------
        w_positive_class = torch.tensor(w_positive_class, device=self.device)
        loss_conf = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits,
            target=ground_truth_label,
            reduction='none',
            pos_weight=w_positive_class)

        # Compute binary cross entropy
        loss_conf = validity_map * loss_conf
        loss_conf = torch.sum(loss_conf) / torch.sum(validity_map)
        #----------------------------fusionnet--------------------------

        loss_info = {
            'loss' : loss_conf
        }

        return loss_conf, loss_info


    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        parameters = \
            list(self.rcnet_encoder.parameters()) +  \
            list(self.rcnet_decoder.parameters()) 

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''
        self.rcnet_encoder.train()
        self.rcnet_decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''
        self.rcnet_encoder.eval()
        self.rcnet_decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.rcnet_encoder.to(device)
        self.rcnet_decoder.to(device)

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {}
        checkpoint['train_step'] = step
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Load weights for encoder, and decoder
        checkpoint['rcnet_encoder_state_dict'] = self.rcnet_encoder.state_dict()
        checkpoint['rcnet_decoder_state_dict'] = self.rcnet_decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore weights for encoder, and decoder
        self.rcnet_encoder.load_state_dict(checkpoint['rcnet_encoder_state_dict'])
        self.rcnet_decoder.load_state_dict(checkpoint['rcnet_decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['train_step'], optimizer
    
    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.rcnet_encoder = torch.nn.DataParallel(self.rcnet_encoder)
        self.rcnet_decoder = torch.nn.DataParallel(self.rcnet_decoder)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image=None,
                    radar_depth=None,
                    input_depth=None,
                    input_response=None,
                    ground_truth=None,
                    ground_truth_label=None,
                    scalars={},
                    n_display=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image : torch.Tensor[float32]
                N x 3 x H x W image
            input_depth : torch.Tensor[float32]
                N x 1 x H x W input depth
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image is not None:
                image_summary = image[0:n_display, ...]

                display_summary_image_text += '_image'
                display_summary_depth_text += '_image'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image_summary.cpu(),
                        torch.zeros_like(image_summary, device=torch.device('cpu'))],
                        dim=-1))
                if display_summary_image[-1].shape[1] == 1:
                    display_summary_depth.append(log_utils.colorize(
                            (display_summary_image[-1] / self.max_predict_depth).cpu(),
                            colormap='viridis'))
                elif display_summary_image[-1].shape[1] == 3:
                    display_summary_depth.append(display_summary_image[-1])


            if radar_depth is not None and input_depth is not None:

                radar_depth_summary = radar_depth[0:n_display, ...]
                input_depth_summary = input_depth[0:n_display, ...]
                
                display_summary_depth_text += '_radar_depth'
                display_summary_depth_text += '_quasi_depth'

                # Add to list of images to log
                radar_depth_summary = log_utils.colorize(
                    (radar_depth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                
                input_depth_summary = log_utils.colorize(
                    (input_depth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')

                display_summary_depth.append(
                    torch.cat([
                        radar_depth_summary,
                        input_depth_summary],
                        dim=3))

                # Log distribution of input depth
                summary_writer.add_histogram(tag + '_radar_depth', radar_depth, global_step=step)
                summary_writer.add_histogram(tag + '_quasi_depth', input_depth, global_step=step)

            if ground_truth is not None:
                ground_truth = ground_truth[0:n_display, ...]
                ground_truth_summary = ground_truth[0:n_display]
                display_summary_depth_text += '_ground_truth'

                n_batch, _, n_height, n_width = ground_truth_summary.shape

                # Add to list of images to log
                ground_truth_summary = log_utils.colorize(
                    (ground_truth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')


                display_summary_depth.append(
                    torch.cat([
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu')),
                        ground_truth_summary
                        ],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)


            if ground_truth_label is not None and input_response is not None:
                ground_truth_label_summary = ground_truth_label[0:n_display, ...]
                response_summary = input_response[0:n_display, ...]

                display_summary_depth_text += '_conf_map'
                display_summary_depth_text += '_ground_truth_label'

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            response_summary.cpu(),
                            colormap='inferno'),
                        log_utils.colorize(
                            ground_truth_label_summary.cpu(),
                            colormap='inferno'),
                             ],
                        dim=3))

                summary_writer.add_histogram(tag + '_ ground_truth_label', ground_truth_label, global_step=step)
                summary_writer.add_histogram(tag + '_conf_map', input_response, global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) > 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                    global_step=step)

            if len(display_summary_depth) > 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                    global_step=step)
