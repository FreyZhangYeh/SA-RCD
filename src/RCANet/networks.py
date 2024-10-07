import torch
from utils import net_utils
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
from utils.data_utils import hist,save_feature_maps,save_feature_maps_batch
from linear_attention import LocalFeatureTransformer
'''
Blocks
'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # shared MLP
        middle_plans = max(in_planes // reduction,1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, middle_plans, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(middle_plans, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)
 
    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result 

class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections
    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetEncoder, self).__init__()

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks2 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks3 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks4 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.blocks5 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.blocks6 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.blocks7 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            self.blocks7 = None

    def _make_layer(self,
                    network_block,
                    n_block,
                    in_channels,
                    out_channels,
                    stride,
                    weight_initializer,
                    activation_func,
                    use_batch_norm):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels : int
                number of channels
            out_channels : int
                number of output channels
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks = []

        for n in range(n_block):

            if n == 0:
                stride = stride
            else:
                in_channels = out_channels
                stride = 1

            block = network_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks.append(block)

        blocks = torch.nn.Sequential(*blocks)

        return blocks

    def forward(self, x):
        '''
        Forward input x through the ResNet model
        Arg(s):
            x : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]

class FusionBlocks(torch.nn.Module):
    def __init__(self,
                 input_channels_image=3,
                 input_channels_depth=2,
                 weight_initializer='kaiming_uniform',
                 use_batch_norm=False,
                 fusion_type='add'
                 ):
        super(FusionBlocks, self).__init__()

        self.input_channels_depth = input_channels_depth
        self.input_channels_image = input_channels_image

        self.weight_initializer = weight_initializer
        self.use_batch_norm = use_batch_norm
        self.fusion_type = fusion_type

        if self.fusion_type == 'add':
            self.conv_project = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
        
        elif self.fusion_type == 'concat':
            self.prjF2I = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
            pass
            
        elif self.fusion_type == 'weight':
            self.conv_weight = net_utils.Conv2d(
                input_channels_depth,
                input_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)
            
        elif self.fusion_type == 'weight_and_project':
            self.conv_weight = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv_project = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
            
        elif self.fusion_type == 'cagf':
            self.conv_weight = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv_project = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
        
        elif self.fusion_type == 'attention':
            #assert self.input_channels_depth == self.input_channels_image,"input_channels_image do not match input_channels_depth"
            self.attention = LocalFeatureTransformer(['self','cross'], n_layers=4, d_model=self.input_channels_image)
            self.prjD2I = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
            self.prjF2I = net_utils.Conv2d(
                input_channels_image + input_channels_image,
                input_channels_image,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

                    
        if 'SGFM' in self.fusion_type:
            self.conv_prjI2D = net_utils.Conv2d(
                input_channels_image,
                input_channels_depth,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
            self.conv_prjD2I = net_utils.Conv2d(
                input_channels_depth*2,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)
            self.conv_att = net_utils.Conv2d(
                input_channels_depth,
                input_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)
            self.conv_weight = net_utils.Conv2d(
                input_channels_depth*2,
                input_channels_depth*2,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)  
            self.conv_F = net_utils.Conv2d(
                input_channels_image*4,
                input_channels_image,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('relu'),
                use_batch_norm=use_batch_norm
                )  
        
        if 'SVFF' in self.fusion_type:
            
            self.conv_dep = net_utils.Conv2d(
                input_channels_image + input_channels_depth,
                input_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('tanh'),
                use_batch_norm=use_batch_norm)
            
            if self.fusion_type == 'SVFF':
                self.conv_img = net_utils.Conv2d(
                    input_channels_image + input_channels_depth,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('tanh'),
                    use_batch_norm=use_batch_norm,
                    bias=True)
                self.conv_F = net_utils.Conv2d(
                    input_channels_depth+input_channels_image,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('gelu'),
                    use_batch_norm=use_batch_norm,
                    bias=True) 
                
            if self.fusion_type == 'SVFF_add':
                self.conv_img = net_utils.Conv2d(
                    input_channels_image + input_channels_depth,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('gelu'),
                    use_batch_norm=use_batch_norm,
                    bias=True)
                self.conv_dep = net_utils.Conv2d(
                    input_channels_image + input_channels_depth,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('gelu'),
                    use_batch_norm=use_batch_norm)
                self.prjD2I=net_utils.Conv2d(
                    input_channels_depth,
                    input_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)

            if self.fusion_type == 'SVFF_SAT':
                self.fusion_CBAM = CBAM(input_channels_depth+input_channels_image)
                self.conv_img = net_utils.Conv2d(
                    input_channels_image + input_channels_depth,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('tanh'),
                    use_batch_norm=use_batch_norm)
                self.conv_F = net_utils.Conv2d(
                    input_channels_depth+input_channels_image,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('gelu'),
                    use_batch_norm=use_batch_norm) 
                
            if self.fusion_type == 'gated_radar_SVFF':
                self.conv_weight = net_utils.Conv2d(
                    input_channels_depth,
                    input_channels_image,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)  
                self.conv_prjD2I = net_utils.Conv2d(
                    input_channels_depth,
                    input_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)
   
        if 'CBAM' in self.fusion_type:

            if self.fusion_type == 'radar_CBAM':
                self.dep_SpatialAttention = SpatialAttention()
                self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
                self.dep_CBAM = CBAM(input_channels_depth)

            elif self.fusion_type == 'cross_di_CBAM':
                # self.dep_SpatialAttention = SpatialAttention()
                # self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
                self.dep_CBAM = CBAM(input_channels_depth)
                self.img_SpatialAttention = SpatialAttention()
                self.img_ChannelAttention = ChannelAttention(input_channels_image)

            elif self.fusion_type == 'cross_di_CBAM_iCBAM':
                self.dep_CBAM = CBAM(input_channels_depth)
                self.img_SpatialAttention = SpatialAttention()
                self.img_ChannelAttention = ChannelAttention(input_channels_image)

            elif self.fusion_type == 'cross_d_CBAM_i_CBAM':
                self.dep_CBAM = CBAM(input_channels_depth)
                self.img_SpatialAttention = SpatialAttention()
                self.img_ChannelAttention = ChannelAttention(input_channels_image)
            
            elif self.fusion_type == 'gated_radar_CBAM':
                self.dep_SpatialAttention = SpatialAttention()
                self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
                self.dep_CBAM = CBAM(input_channels_depth)
                self.conv_weight = net_utils.Conv2d(
                    input_channels_depth,
                    input_channels_depth,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)

            elif self.fusion_type == 'gated_cross_CBAM':
                self.dep_SpatialAttention = SpatialAttention()
                self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
                self.dep_CBAM = CBAM(input_channels_depth)
                self.img_SpatialAttention = SpatialAttention()
                self.img_ChannelAttention = ChannelAttention(input_channels_image)
                self.conv_weight = net_utils.Conv2d(
                    input_channels_depth,
                    input_channels_depth,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)
                
            elif self.fusion_type == 'concat_CBAM':
                self.fusion_CBAM = CBAM(input_channels_depth + input_channels_image)
                self.prjF2I = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_image,
                kernel_size=7,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

            elif self.fusion_type == 'cross_id_CBAM':
                self.dep_SpatialAttention = SpatialAttention()
                self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
                self.dep_CBAM = CBAM(input_channels_depth)
                self.img_SpatialAttention = SpatialAttention()
                self.img_ChannelAttention = ChannelAttention(input_channels_image)
                
            elif self.fusion_type == 'cross_di_concat_CBAM':
                self.fusion_CBAM = CBAM(input_channels_depth + input_channels_image)
                self.dep_CBAM = CBAM(input_channels_depth)
                self.img_SpatialAttention = SpatialAttention()
                self.img_ChannelAttention = ChannelAttention(input_channels_image)
                self.prjF2I = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_image,
                kernel_size=7,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

            if self.input_channels_depth != self.input_channels_image and 'concat' not in fusion_type:
                self.conv_prjD2I = net_utils.Conv2d(
                input_channels_depth,
                input_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

    def forward(self, conv_image=None, conv_depth=None, conf_map=None):

        if self.fusion_type == 'add':
            if self.input_channels_depth == self.input_channels_image:
                fusionblocks = conv_depth + conv_image
            else:
                fusionblocks = self.conv_project(conv_depth) + conv_image
        
        elif self.fusion_type == 'concat':
            fusion = torch.cat([conv_depth, conv_image], dim=1)
            fusionblocks = self.prjF2I(fusion)
            fusionblocks = fusion

        elif self.fusion_type == 'weight':
            fusionblocks = self.conv_weight(conv_depth)*conv_depth + conv_image

        elif self.fusion_type == 'weight_and_project':
            weight = self.conv_weight(conv_depth)
            if self.input_channels_depth == self.input_channels_image:
                fusionblocks = weight*conv_depth + conv_image
            else:
                project = self.conv_project(conv_depth)
                fusionblocks = weight*project + conv_image

        elif self.fusion_type == 'cagf':
            weight = self.conv_weight(conv_depth)
            conf_map_gap = F.adaptive_avg_pool2d(conf_map, output_size=(1, 1))
            if self.input_channels_depth == self.input_channels_image:
                fusionblocks = weight*conv_depth*conf_map_gap + conv_image
            else:
                project = self.conv_project(conv_depth)
                fusionblocks = weight*project*conf_map_gap + conv_image
            
        elif self.fusion_type == 'attention':
            conv_depth = self.prjD2I(conv_depth)
            latent_image = conv_image.view(conv_image.shape[0],conv_image.shape[1], -1).permute(0, 2, 1)
            latent_depth = conv_depth.view(conv_depth.shape[0],conv_depth.shape[1], -1).permute(0, 2, 1)
            latent_depth_tf, latent_image_tf = self.attention(latent_depth, latent_image)
            latent_depth_tf = latent_depth_tf.permute(0, 2, 1).view(conv_depth.shape)
            latent_image_tf = latent_image_tf.permute(0, 2, 1).view(conv_image.shape)
            latent_fusion = torch.cat([latent_image_tf, latent_depth_tf], dim=1)
            latent_ouput = self.prjF2I(latent_fusion)
            fusionblocks = latent_ouput

        elif self.fusion_type == 'image_only':
            fusionblocks = conv_image

        if 'SVFF' in self.fusion_type:
            # if self.fusion_type == 'SVFF':
            #     feature = torch.cat([conv_depth, conv_image], dim=1)
            #     d_map = self.conv_dep(feature)
            #     i_map = self.conv_img(feature)
            #     fusion = self.conv_prjD2I(conv_depth*d_map) + conv_image*i_map
            #     fusionblocks = self.conv_F(fusion)

            if self.fusion_type == 'SVFF':
                feature = torch.cat([conv_depth, conv_image], dim=1)
                d_map = self.conv_dep(feature)
                i_map = self.conv_img(feature)
                fusion = torch.cat([conv_depth*d_map,conv_image*i_map], dim=1)
                fusionblocks = self.conv_F(fusion)


            if self.fusion_type == 'SVFF_add':
                feature = torch.cat([conv_depth, conv_image], dim=1)
                d_prjed = self.prjD2I(conv_depth)
                d_map = self.conv_dep(feature)
                i_map = self.conv_img(feature)
                fusionblocks = d_prjed*d_map + conv_image*i_map

            elif self.fusion_type == 'SVFF_SAT':
                feature = torch.cat([conv_depth, conv_image], dim=1)
                d_map = self.conv_dep(feature)
                i_map = self.conv_img(feature)
                fusion = torch.cat([conv_depth*d_map,conv_image*i_map], dim=1)
                fusion_atted = self.fusion_CBAM(fusion) + fusion
                fusionblocks = self.conv_F(fusion_atted)

            elif self.fusion_type == 'gated_radar_SVFF':
                feature = torch.cat([conv_depth, conv_image], dim=1)
                d_map = self.conv_dep(feature)
                d_weight = self.conv_weight(conv_depth*d_map)
                d_project = self.conv_prjD2I(conv_depth*d_map)
                fusionblocks = d_project*d_weight + conv_image

        if 'CBAM' in self.fusion_type:
            if self.fusion_type == 'radar_CBAM':
                dep_cbam_map = self.dep_CBAM(conv_depth)
                dep_atted = dep_cbam_map + conv_depth 
                if self.input_channels_depth == self.input_channels_image:
                    fusionblocks = conv_image + dep_atted
                else:
                    fusionblocks = conv_image + self.conv_prjD2I(dep_atted)

            elif self.fusion_type == 'cross_di_CBAM':
                dep_cbam_map = self.dep_CBAM(conv_depth)
                img_channel_atted_map = self.img_ChannelAttention(conv_image)
                img_spatial_input = img_channel_atted_map*conv_image
                img_spatial_att_map = self.img_SpatialAttention(img_spatial_input)
                dep_atted = conv_depth + dep_cbam_map*img_spatial_att_map
                if self.input_channels_depth == self.input_channels_image:
                    fusionblocks = conv_image + dep_atted
                else:
                    fusionblocks = conv_image + self.conv_prjD2I(dep_atted)

            elif self.fusion_type == 'cross_di_CBAM_iCBAM':
                dep_cbam_map = self.dep_CBAM(conv_depth)
                img_channel_atted_map = self.img_ChannelAttention(conv_image)
                img_spatial_input = img_channel_atted_map*conv_image
                img_spatial_att_map = self.img_SpatialAttention(img_spatial_input)
                dep_atted = conv_depth + dep_cbam_map*img_spatial_att_map
                img_atted = conv_image + img_spatial_input*img_spatial_att_map
                if self.input_channels_depth == self.input_channels_image:
                    fusionblocks = img_atted + dep_atted
                else:
                    fusionblocks = img_atted + self.conv_prjD2I(dep_atted)

            elif self.fusion_type == 'concat_CBAM':
                fusion = torch.cat([conv_depth, conv_image], dim=1)
                fusion_atted = fusion + self.fusion_CBAM(fusion)
                fusionblocks = self.prjF2I(fusion_atted)

            elif self.fusion_type == 'cross_id_CBAM':
                img_channel_atted_map = self.img_ChannelAttention(conv_image)
                img_spatial_input = img_channel_atted_map*conv_image
                img_spatial_att_map = self.img_SpatialAttention(img_spatial_input)
                dep_denoised = conv_depth*img_spatial_att_map
                dep_atted = conv_depth + self.dep_CBAM(dep_denoised)
                if self.input_channels_depth == self.input_channels_image:
                    fusionblocks = conv_image + dep_atted
                else:
                    fusionblocks = conv_image + self.conv_prjD2I(dep_atted)

            elif self.fusion_type == 'cross_di_concat_CBAM':
                dep_cbam_map = self.dep_CBAM(conv_depth)
                img_channel_atted_map = self.img_ChannelAttention(conv_image)
                img_spatial_input = img_channel_atted_map*conv_image
                img_spatial_att_map = self.img_SpatialAttention(img_spatial_input)
                dep_atted = conv_depth + dep_cbam_map*img_spatial_att_map
                fusion = torch.cat([dep_atted, conv_image], dim=1)
                fusion_atted = fusion + self.fusion_CBAM(fusion)
                fusionblocks = self.prjF2I(fusion)

            elif self.fusion_type == 'cross_d_CBAM_i_CBAM':
                dep_cbam_map = self.dep_CBAM(conv_depth)
                img_channel_atted_map = self.img_ChannelAttention(conv_image)
                img_spatial_input = img_channel_atted_map*conv_image
                img_spatial_att_map = self.img_SpatialAttention(img_spatial_input)
                dep_atted = (conv_depth + dep_cbam_map)*img_spatial_att_map
                if self.input_channels_depth == self.input_channels_image:
                    fusionblocks = conv_image + dep_atted
                else:
                    fusionblocks = conv_image + self.conv_prjD2I(dep_atted)
        
        return fusionblocks

class GuidanceBlocks(torch.nn.Module):
    def __init__(self,
                input_channels_image=3,
                input_channels_depth=2,
                weight_initializer='kaiming_uniform',
                kernel_size = 7,
                use_batch_norm=False,
                activation_func='relu',
                guidance_type = 'concat'
                ):
        super(GuidanceBlocks, self).__init__()
        
        self.guidance_type = guidance_type
        self.input_channels_image = input_channels_image
        self.input_channels_depth = input_channels_depth
        
        if guidance_type == 'concat':
            self.guidance_conv = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_depth,
                kernel_size=kernel_size,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm
            )
        elif guidance_type == 'CBAM':
            self.CBAM = CBAM(input_channels_depth + input_channels_image)
            self.guidance_conv = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_depth,
                kernel_size=kernel_size,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm
            )
        elif guidance_type == 'RGI_CBAM':
            self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
            self.dep_SpatialAttention = SpatialAttention()
            self.guidance_conv = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_depth,
                kernel_size=kernel_size,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm
            )
            # self.prjD2I = net_utils.Conv2d(
            #     input_channels_depth,
            #     input_channels_image,
            #     kernel_size=1,
            #     stride=1,
            #     weight_initializer=weight_initializer,
            #     activation_func=net_utils.activation_func('linear'),
            #     use_batch_norm=use_batch_norm)
        elif guidance_type == 'RGI_CBAM_withoutr_add':
            self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
            self.dep_SpatialAttention = SpatialAttention()
            self.prjI2D = net_utils.Conv2d(
                input_channels_image,
                input_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif guidance_type == 'RGI_CBAM_concat':
            self.dep_ChannelAttention = ChannelAttention(input_channels_depth)
            self.dep_SpatialAttention = SpatialAttention()
            self.guidance_conv = net_utils.Conv2d(
                input_channels_depth + input_channels_image,
                input_channels_depth,
                kernel_size=kernel_size,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm
            )
        
    def forward(self,conv_image=None, conv_depth=None):
        if self.guidance_type == 'concat':
            guidance = torch.cat([conv_depth, conv_image], dim=1)
            conv_depth = self.guidance_conv(guidance)
            
        elif self.guidance_type == 'CBAM':
            guidance = torch.cat([conv_depth, conv_image], dim=1)
            atted_guidance = self.CBAM(guidance) + guidance
            conv_depth = self.guidance_conv(atted_guidance)
            
        elif self.guidance_type == 'RGI_CBAM':
            dep_ca_map = self.dep_ChannelAttention(conv_depth)
            dep_sa_map = self.dep_SpatialAttention(conv_depth*dep_ca_map)
            reversed_dep_sa_map = 1.0 - dep_sa_map
            #reversed_dep_sa_map = self.prjD2I(reversed_dep_sa_map)
            guidance = torch.cat([conv_depth, conv_image + conv_image*reversed_dep_sa_map], dim=1)
            conv_depth = self.guidance_conv(guidance)
            
        elif self.guidance_type == 'RGI_CBAM_withoutr_add':
            dep_ca_map = self.dep_ChannelAttention(conv_depth)
            dep_sa_map = self.dep_SpatialAttention(conv_depth*dep_ca_map)
            reversed_dep_sa_map = 1.0 - dep_sa_map
            selected_image = conv_image*reversed_dep_sa_map
            conv_depth = conv_depth + self.prjI2D(selected_image)
            
        elif self.guidance_type == 'RGI_CBAM_concat':
            dep_ca_map = self.dep_ChannelAttention(conv_depth)
            dep_sa_map = self.dep_SpatialAttention(conv_depth*dep_ca_map)
            reversed_dep_sa_map = 1.0 - dep_sa_map
            #reversed_dep_sa_map = self.prjD2I(reversed_dep_sa_map)
            guidance = torch.cat([conv_depth, conv_image*reversed_dep_sa_map], dim=1)
            conv_depth = self.guidance_conv(guidance)
            
        elif self.guidance_type == None:
            conv_depth = conv_depth
            
        return conv_depth

class OutputConv(nn.Module):
    """Output conv block.
    """

    def __init__(self, features, groups=1):

        super(OutputConv, self).__init__()


        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.LeakyReLU(negative_slope=0.10, inplace=True),
            nn.Conv2d(features//2, features//4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.10, inplace=True),
            nn.Conv2d(features//4, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):  
        logits = self.output_conv(x)  
        return logits

'''
Encoders
'''
class RCNetEncoder(torch.nn.Module):
    '''
    FusionNet encoder with skip connections
    Arg(s):
        n_layer : int
            number of layer for encoder
        input_channels_image : int
            number of channels in input data
        input_channels_depth : int
            number of channels in input data
        n_filters_per_block : list[int]
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        fusion_type : str
            add, weight
    '''

    def __init__(self,
                 n_img_layer=34,
                 n_dep_layer=18,
                 input_channels_image=3,
                 input_channels_depth=2,
                 n_filters_encoder_image=[32, 64, 128, 256, 256],
                 n_filters_encoder_depth=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 dropout_prob=0.1,
                 fusion_type='add',
                 guidance = 'concat',
                 guidance_layers = [3,4,5,6],
                 fusion_layers = [1,2],
                 device='cuda'):
        super(RCNetEncoder, self).__init__()

        self.fusion_type = fusion_type
        self.weight_initializer = weight_initializer
        self.use_batch_norm = use_batch_norm
        self.device = device
        self.guidance = guidance
        self.guidance_layers = guidance_layers
        self.dropout_prob = dropout_prob

        # define img encoder
        if n_img_layer == 18:
            img_resnet_n_blocks = [2, 2, 2, 2]
        elif n_img_layer == 34:
            img_resnet_n_blocks = [3, 4, 6, 3]
        else:
            raise ValueError('Only supports 18, 34 layer architecture')
        
        # define dep encoder
        if n_dep_layer == 18:
            dep_resnet_n_blocks = [2, 2, 2, 2]
        elif n_dep_layer == 34:
            dep_resnet_n_blocks = [3, 4, 6, 3]
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        resnet_block = net_utils.ResNetBlock

        assert len(n_filters_encoder_image) == len(n_filters_encoder_depth)

        for n in range(len(n_filters_encoder_image) - len(img_resnet_n_blocks) - 1):
            img_resnet_n_blocks = img_resnet_n_blocks + [img_resnet_n_blocks[-1]]
            dep_resnet_n_blocks = dep_resnet_n_blocks + [dep_resnet_n_blocks[-1]]


        network_depth = len(n_filters_encoder_image)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(img_resnet_n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)
        
        if self.guidance and (filter_idx+1) in guidance_layers:
            self.guidanceblock1 = GuidanceBlocks(
                input_channels_image = input_channels_image,
                input_channels_depth = input_channels_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                guidance_type=guidance)
        
        # Resolution 1/1 -> 1/2
        self.conv1_image = net_utils.Conv2d(
            input_channels_image,
            n_filters_encoder_image[filter_idx],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv1_depth = net_utils.Conv2d(
            input_channels_depth,
            n_filters_encoder_depth[filter_idx],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fusionblock1 = FusionBlocks(
             input_channels_image = n_filters_encoder_image[filter_idx],
             input_channels_depth = n_filters_encoder_depth[filter_idx],
             weight_initializer = weight_initializer,
             use_batch_norm = use_batch_norm,
             fusion_type = fusion_layers[filter_idx] 
        )
            
        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]
        
        if self.guidance and (filter_idx+1) in guidance_layers:
            self.guidanceblock2 = GuidanceBlocks(
                input_channels_image = in_channels_image,
                input_channels_depth = in_channels_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                guidance_type=guidance) 

        self.blocks2_image, self.blocks2_depth = self._make_layer(
            network_block=resnet_block,
            n_block_img=img_resnet_n_blocks[block_idx],
            n_block_dep=dep_resnet_n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
 
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fusionblock2 = FusionBlocks(
             input_channels_image = out_channels_image,
             input_channels_depth = out_channels_depth,
             weight_initializer = weight_initializer,
             use_batch_norm = use_batch_norm,
             fusion_type = fusion_layers[filter_idx]
        )

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]
        
        if self.guidance and (filter_idx+1) in guidance_layers:
            self.guidanceblock3 = GuidanceBlocks(
                input_channels_image = in_channels_image,
                input_channels_depth = in_channels_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                guidance_type=guidance)

        self.blocks3_image, self.blocks3_depth = self._make_layer(
            network_block=resnet_block,
            n_block_img=img_resnet_n_blocks[block_idx],
            n_block_dep=dep_resnet_n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        
        self.dropout3 = nn.Dropout(p=dropout_prob)

        self.fusionblock3 = FusionBlocks(
             input_channels_image = out_channels_image,
             input_channels_depth = out_channels_depth,
             weight_initializer = weight_initializer,
             use_batch_norm = use_batch_norm,
             fusion_type = fusion_layers[filter_idx]
        )

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]
        
        if self.guidance and (filter_idx+1) in guidance_layers:
            self.guidanceblock4 = GuidanceBlocks(
                input_channels_image = in_channels_image,
                input_channels_depth = in_channels_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                guidance_type=guidance)

        self.blocks4_image, self.blocks4_depth = self._make_layer(
            network_block=resnet_block,
            n_block_img=img_resnet_n_blocks[block_idx],
            n_block_dep=dep_resnet_n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.fusionblock4 = FusionBlocks(
             input_channels_image = out_channels_image,
             input_channels_depth = out_channels_depth,
             weight_initializer = weight_initializer,
             use_batch_norm = use_batch_norm,
             fusion_type = fusion_layers[filter_idx]
        )

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]
        
        if self.guidance and (filter_idx+1) in guidance_layers:
            self.guidanceblock5 = GuidanceBlocks(
                input_channels_image = in_channels_image,
                input_channels_depth = in_channels_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                guidance_type=guidance)

        self.blocks5_image, self.blocks5_depth = self._make_layer(
            network_block=resnet_block,
            n_block_img=img_resnet_n_blocks[block_idx],
            n_block_dep=dep_resnet_n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        
        self.dropout5 = nn.Dropout(p=dropout_prob)

        self.fusionblock5 = FusionBlocks(
             input_channels_image = out_channels_image,
             input_channels_depth = out_channels_depth,
             weight_initializer = weight_initializer,
             use_batch_norm = use_batch_norm,
             fusion_type = fusion_layers[filter_idx]
        )


        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters_encoder_image):

            in_channels_image, out_channels_image = [
                n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
            ]

            in_channels_depth, out_channels_depth = [
                n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
            ]
            
            if self.guidance and (filter_idx+1) in guidance_layers:
                self.guidanceblock6 = GuidanceBlocks(
                    input_channels_image = in_channels_image,
                    input_channels_depth = in_channels_depth,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    guidance_type=guidance)

            self.blocks6_image, self.blocks6_depth = self._make_layer(
                network_block=resnet_block,
                n_block_img=img_resnet_n_blocks[block_idx],
                n_block_dep=dep_resnet_n_blocks[block_idx],
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                out_channels_image=out_channels_image,
                out_channels_depth=out_channels_depth,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            self.dropout6 = nn.Dropout(p=dropout_prob)

            self.fusionblock6 = FusionBlocks(
             input_channels_image = out_channels_image,
             input_channels_depth = out_channels_depth,
             weight_initializer = weight_initializer,
             use_batch_norm = use_batch_norm,
             fusion_type = fusion_layers[filter_idx]
        )

            
        else:
            self.blocks6_image = None
            self.blocks6_depth = None
            self.fusionblock6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters_encoder_image):

            in_channels_image, out_channels_image = [
                n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
            ]

            in_channels_depth, out_channels_depth = [
                n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
            ]
            
            if self.guidance and guidance_layers > 0:
                self.guidanceblock7 = GuidanceBlocks(
                    input_channels_image = in_channels_image,
                    input_channels_depth = in_channels_depth,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    guidance_type=guidance)
                guidance_layers -= 1

            self.blocks7_image, self.blocks7_depth = self._make_layer(
                network_block=resnet_block,
                n_block_img=img_resnet_n_blocks[block_idx],
                n_block_dep=dep_resnet_n_blocks[block_idx],
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                out_channels_image=out_channels_image,
                out_channels_depth=out_channels_depth,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
            
            self.dropout7 = nn.Dropout(p=dropout_prob)
            
            self.fusionblock7 = FusionBlocks(
                input_channels_image = out_channels_image,
                input_channels_depth = out_channels_depth,
                weight_initializer = weight_initializer,
                use_batch_norm = use_batch_norm,
                fusion_type = fusion_layers[filter_idx]
        )

            
        else:
            self.blocks7_image = None
            self.blocks7_depth = None
            self.fusionblock7 = None

        self.module_names = [name for name, _ in self.named_modules()]

    def _make_layer(self,
                    network_block,
                    n_block_img,
                    n_block_dep,
                    in_channels_image,
                    in_channels_depth,
                    out_channels_image,
                    out_channels_depth,
                    stride,
                    weight_initializer,
                    activation_func,
                    use_batch_norm):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels_image : int
                number of channels in image branch
            in_channels_depth : int
                number of channels in depth branch
            out_channels_image : int
                number of output channels in image branch
            out_channels_depth : int
                number of output channels in depth branch
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks_image = []
        blocks_depth = []

        for n in range(n_block_img):

            if n == 0:
                stride_img = stride
            else:
                in_channels_image = out_channels_image
                stride_img = 1

            block_image = network_block(
                in_channels=in_channels_image,
                out_channels=out_channels_image,
                stride=stride_img,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks_image.append(block_image)


        for n in range(n_block_dep):

            if n == 0:
                stride_dep = stride
            else:
                in_channels_depth = out_channels_depth
                stride_dep = 1

            block_depth = network_block(
                in_channels=in_channels_depth,
                out_channels=out_channels_depth,
                stride=stride_dep,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks_depth.append(block_depth)


        blocks_image = torch.nn.Sequential(*blocks_image)
        blocks_depth = torch.nn.Sequential(*blocks_depth)

        return blocks_image, blocks_depth

    def forward(self, image, depth):
        '''
        Forward input x through the ResNet model
        Arg(s):
            image : torch.Tensor
            depth : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''
        layers = []
        
        # Resolution 1/1 -> 1/2
        conv1_image = self.conv1_image(image)
        conv1_image = self.dropout1(conv1_image)
        depth = self.guidanceblock1(image,depth) if "guidanceblock1" in self.module_names else depth
        conv1_depth = self.conv1_depth(depth)
        conv1_depth = self.dropout1(conv1_depth)

        conv1 = self.fusionblock1(          
            conv_image = conv1_image,
            conv_depth = conv1_depth,)
        
        layers.append(conv1)

        # Resolution 1/2 -> 1/4
        max_pool_image = self.max_pool(conv1_image)
        max_pool_depth = self.max_pool(conv1_depth)

        blocks2_image = self.blocks2_image(max_pool_image)
        blocks2_image = self.dropout2(blocks2_image)
        max_pool_depth = self.guidanceblock2(max_pool_image,max_pool_depth) if "guidanceblock2" in self.module_names else max_pool_depth
        blocks2_depth = self.blocks2_depth(max_pool_depth)
        blocks2_depth = self.dropout2(blocks2_depth)

        blocks2 = self.fusionblock2(
            blocks2_image,
            blocks2_depth
        )

        layers.append(blocks2)

        # Resolution 1/4 -> 1/8
        blocks3_image = self.blocks3_image(blocks2_image)
        blocks3_image = self.dropout3(blocks3_image)
        blocks2_depth = self.guidanceblock3(blocks2_image,blocks2_depth) if "guidanceblock3" in self.module_names else blocks2_depth
        blocks3_depth = self.blocks3_depth(blocks2_depth)
        blocks3_depth = self.dropout3(blocks3_depth)

        blocks3 = self.fusionblock3(
            blocks3_image,
            blocks3_depth
        )

        layers.append(blocks3)

        # Resolution 1/8 -> 1/16
        blocks4_image = self.blocks4_image(blocks3_image)
        blocks4_image = self.dropout4(blocks4_image)
        blocks3_depth = self.guidanceblock4(blocks3_image,blocks3_depth) if "guidanceblock4" in self.module_names else blocks3_depth
        blocks4_depth = self.blocks4_depth(blocks3_depth)
        blocks4_depth = self.dropout4(blocks4_depth)

        blocks4 = self.fusionblock4(
            blocks4_image,
            blocks4_depth
        )

        layers.append(blocks4)

        # Resolution 1/16 -> 1/32
        blocks5_image = self.blocks5_image(blocks4_image)
        blocks5_image = self.dropout5(blocks5_image) 
        blocks4_depth = self.guidanceblock5(blocks4_image,blocks4_depth) if "guidanceblock5" in self.module_names else blocks4_depth
        blocks5_depth = self.blocks5_depth(blocks4_depth)
        blocks5_depth = self.dropout5(blocks5_depth)

        blocks5 = self.fusionblock5(
            blocks5_image,
            blocks5_depth
        )

        layers.append(blocks5)

        # Resolution 1/32 -> 1/64
        if self.blocks6_image is not None and self.blocks6_depth is not None:
            blocks6_image = self.blocks6_image(blocks5_image)
            blocks6_image = self.dropout6(blocks6_image)
            blocks5_depth = self.guidanceblock6(blocks5_image,blocks5_depth) if "guidanceblock6" in self.module_names else blocks5_depth
            blocks6_depth = self.blocks6_depth(blocks5_depth)
            blocks6_depth = self.dropout6(blocks6_depth)
            
            blocks6 = self.fusionblock6(
                blocks6_image,
                blocks6_depth
            )

            layers.append(blocks6)

        # Resolution 1/64 -> 1/128
        if self.blocks7_image is not None and self.blocks7_depth is not None:
            blocks7_image = self.blocks7_image(blocks6_image)
            blocks7_image = self.dropout7(blocks7_image) 
            blocks6_depth = self.guidanceblock7(blocks6_image,blocks6_depth) if "guidanceblock7" in self.module_names else blocks6_depth
            blocks7_depth = self.blocks7_depth(blocks6_depth)
            blocks7_depth = self.dropout7(blocks7_depth)

            blocks7 = self.fusionblock2(
                blocks7_image,
                blocks7_depth
            )

            layers.append(blocks7)

        return layers[-1], layers[:-1]

'''
Decoder Architectures
'''
        
class RCNetDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections
    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : int list
            number of filters to use at each decoder block
        n_skips : int list
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=1,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 dropout_prob=0.1,
                 deconv_type='up'
                 ):
        super(RCNetDecoder, self).__init__()

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func
        self.dropout_prob = dropout_prob

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2

        filter_idx = 0

        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[filter_idx], n_filters[filter_idx]
        ]

        # Resolution 1/128 -> 1/64
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)
            
            self.dropout6 = nn.Dropout(p=dropout_prob)
            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv6 = None

        # Resolution 1/64 -> 1/32
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)
            
            self.dropout5 = nn.Dropout(p=dropout_prob)
            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv5 = None

        # Resolution 1/32 -> 1/16
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)
        
        self.dropout4 = nn.Dropout(p=dropout_prob)
        # Resolution 1/16 -> 1/8
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)
        
        self.dropout3 = nn.Dropout(p=dropout_prob)

        if self.n_resolution > 3:
            self.output3 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/8 -> 1/4
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 3:
            skip_channels = skip_channels + output_channels

        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)
        
        self.dropout2 = nn.Dropout(p=dropout_prob)

        if self.n_resolution > 2:
            self.output2 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/4 -> 1/2
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 2:
            skip_channels = skip_channels + output_channels

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)
        
        self.dropout1 = nn.Dropout(p=dropout_prob) 

        if self.n_resolution > 1:
            self.output1 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/2 -> 1/1
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)
        
        self.dropout0 = nn.Dropout(p=dropout_prob)

        self.conf_map_output = OutputConv(
                features = out_channels)
            


    def forward(self, x, skips, shape=None):
        '''
        Forward latent vector x through decoder network
        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
            shape : tuple[int]
                (height, width) tuple denoting output size
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Start at the end and walk backwards through skip connections
        n = len(skips) - 1

        # Resolution 1/128 -> 1/64
        if self.deconv6 is not None:
            layers.append(self.dropout6(self.deconv6(layers[-1], skips[n])))
            n = n - 1

        # Resolution 1/64 -> 1/32
        if self.deconv5 is not None:
            layers.append(self.dropout5(self.deconv5(layers[-1], skips[n])))
            n = n - 1

        # Resolution 1/32 -> 1/16
        layers.append(self.dropout4(self.deconv4(layers[-1], skips[n])))

        # Resolution 1/16 -> 1/8
        n = n - 1

        layers.append(self.dropout3(self.deconv3(layers[-1], skips[n])))

        if self.n_resolution > 3:
            output3 = self.output3(layers[-1])
            outputs.append(output3)

            upsample_output3 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/8 -> 1/4
        n = n - 1

        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
        layers.append(self.dropout2(self.deconv2(layers[-1], skip)))

        if self.n_resolution > 2:
            output2 = self.output2(layers[-1])
            outputs.append(output2)

            upsample_output2 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/4 -> 1/2
        n = n - 1

        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        layers.append(self.dropout1(self.deconv1(layers[-1], skip)))

        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)

            upsample_output1 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                # If there is skip connection at layer 0
                skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                layers.append(self.dropout0(self.deconv0(layers[-1], skip)))
            else:

                if n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1], shape=shape[-2:]))
            
            conf_map_logits = self.conf_map_output(layers[-1])
            outputs.append(conf_map_logits)


            return outputs,conf_map_logits


