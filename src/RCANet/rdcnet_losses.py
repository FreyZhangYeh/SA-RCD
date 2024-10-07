import torch
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
from math import exp
import torch.nn.functional as F

def save_feature_maps(feature_maps, file_name, fusion_type):
    directory = os.path.join("/home/zfy/radar-camera-fusion-depth/visl/feature_map",fusion_type)
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Reshape feature maps from [1, c, h, w] to [c, 1, h, w]
    feature_maps = feature_maps.squeeze(0).unsqueeze(1)

    # Normalize each feature map
    normalized_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())

    # Construct the full file path
    file_path = os.path.join(directory, file_name)

    # Save the grid of images
    torchvision.utils.save_image(normalized_maps, file_path, nrow=4)  # Adjust nrow to your preference
    
def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle

class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        #show_grad(grad_pred[0],grad_gt[0])

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])

        if not return_interpolated:
            return loss
        return loss, intr_input
    
    def get_grad_map(self,input,target):

        grad_pred = grad(input)[0]
        grad_gt = grad(target)[0]
        
        grad_pred = grad_pred/grad_pred.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        grad_gt = grad_gt/grad_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]


        grad_pred = nn.functional.interpolate(
                grad_pred, input.shape[-2:], mode='bilinear', align_corners=True)
        grad_gt = nn.functional.interpolate(
                grad_gt, target.shape[-2:], mode='bilinear', align_corners=True)
        

        return grad_pred,grad_gt


    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
    
def create_window(window_size, channel, mu=1.5):
    _1D_window = gaussian(window_size, mu).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window


def create_window_avg(window_size, channel):
    _2D_window = torch.ones(window_size, window_size).float().unsqueeze(0).unsqueeze(0) / (window_size ** 2)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

class SSIM(torch.nn.Module):
    def __init__(self, window_size=5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel)
    
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def get_grad_map(self,pred,gt):
        with torch.no_grad():
            batch_size = pred.shape[0]
            
            pred_grad = self.edge_conv(pred) 
            pred_grad = pred_grad.contiguous().view(-1, 2, pred.size(2), pred.size(3))

            gt_grad = self.edge_conv(gt) 
            gt_grad = gt_grad.contiguous().view(-1, 2, gt.size(2), gt.size(3))
            
            pred_edge_np_list = []
            gt_edge_np_list = []

            for i in range(batch_size):
                pred_dx = pred_grad[i, 0, :, :].contiguous().view_as(pred[i])
                pred_dy = pred_grad[i, 1, :, :].contiguous().view_as(pred[i])
                gt_dx = gt_grad[i, 0, :, :].contiguous().view_as(gt[i])
                gt_dy = gt_grad [i, 1, :, :].contiguous().view_as(gt[i])

                pred_edge = torch.sqrt(pred_dx ** 2 + pred_dy ** 2)
                gt_edge = torch.sqrt(gt_dx ** 2 + gt_dy ** 2)

                pred_edge_np = pred_edge
                pred_edge_np = (pred_edge_np - pred_edge_np.min()) / (pred_edge_np.max() - pred_edge_np.min())
                gt_edge_np = gt_edge
                gt_edge_np = (gt_edge_np - gt_edge_np.min()) / (gt_edge_np.max() - gt_edge_np.min())
                
                pred_edge_np_list.append(pred_edge_np)
                gt_edge_np_list.append(gt_edge_np)
                
        return torch.stack(pred_edge_np_list,dim=0),torch.stack(gt_edge_np_list,dim=0)
        

    def forward(self, pred, gt):
        pred_out = self.edge_conv(pred) 
        pred_out = pred_out.contiguous().view(-1, 2, pred.size(2), pred.size(3))

        gt_out = self.edge_conv(gt) 
        gt_out = gt_out.contiguous().view(-1, 2, gt.size(2), gt.size(3))

        pred_grad = pred_out
        gt_grad = gt_out

        #show_sobel(pred_grad,gt_grad,pred,gt)

        pred_grad_dx = pred_grad[:, 0, :, :].contiguous().view_as(pred)
        pred_grad_dy = pred_grad[:, 1, :, :].contiguous().view_as(pred)
        gt_grad_dx = gt_grad[:, 0, :, :].contiguous().view_as(gt)
        gt_grad_dy = gt_grad[:, 1, :, :].contiguous().view_as(gt)

        loss_dx = torch.log(torch.abs(pred_grad_dx - gt_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(pred_grad_dy - gt_grad_dy) + 0.5).mean()


  
        return loss_dx + loss_dy
    

def show_sobel(output_grad,depth_grad,pred,gt):

    with torch.no_grad():
        batch_size = depth_grad.shape[0]

        for i in range(batch_size):
            # 计算每个方向的梯度
            depth_grad_dx = depth_grad[i, 0, :, :].contiguous().view_as(gt[i])
            depth_grad_dy = depth_grad[i, 1, :, :].contiguous().view_as(gt[i])
            output_grad_dx = output_grad[i, 0, :, :].contiguous().view_as(pred[i])
            output_grad_dy = output_grad[i, 1, :, :].contiguous().view_as(pred[i])

            # 计算总的边缘强度
            depth_edge = torch.sqrt(depth_grad_dx ** 2 + depth_grad_dy ** 2)
            output_edge = torch.sqrt(output_grad_dx ** 2 + output_grad_dy ** 2)

            # 归一化边缘强度
            depth_edge_np = depth_edge.cpu().numpy()
            depth_edge_np = (depth_edge_np - depth_edge_np.min()) / (depth_edge_np.max() - depth_edge_np.min())
            output_edge_np = output_edge.cpu().numpy()
            output_edge_np = (output_edge_np - output_edge_np.min()) / (output_edge_np.max() - output_edge_np.min())

            # 合并 pred 和 gt 的边缘图像
            combined_edge = np.concatenate((depth_edge_np, output_edge_np), axis=2).squeeze()

            # 保存图像
            plt.imsave(os.path.join("visl/gradl1/" + f'combined_edge_{i}.png'), combined_edge, cmap='gray')

def show_grad(pred_edge,gt_edge,save_dir):
    
    os.makedirs(save_dir,exist_ok=True)

    with torch.no_grad():

        batch_size = pred_edge.shape[0]
        

        for i in range(batch_size):
            # 计算每个方向的梯度


            # 归一化边缘强度
            pred_edge_np = pred_edge[i].cpu().numpy()
            pred_edge_np = (pred_edge_np - pred_edge_np.min()) / (pred_edge_np.max() - pred_edge_np.min())
            gt_edge_np = gt_edge[i].cpu().numpy()
            gt_edge_np = (gt_edge_np - gt_edge_np.min()) / (gt_edge_np.max() - gt_edge_np.min())

            # 合并 pred 和 gt 的边缘图像
            combined_edge = np.concatenate((gt_edge_np,pred_edge_np), axis=2).squeeze()

            # 保存图像
            plt.imsave(os.path.join(save_dir, f'combined_edge_{i}.png'), combined_edge, cmap='gray')

def smooth_l1_loss(src, tgt):
    '''
    Computes smooth_l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean smooth l1 loss across batch
    '''

    return torch.nn.functional.smooth_l1_loss(src, tgt, reduction='mean')

def l1_loss(src, tgt):
    '''
    Computes l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean l1 loss across batch
    '''

    return torch.nn.functional.l1_loss(src, tgt, reduction='mean')

def l2_loss(src, tgt):
    '''
    Computes l2 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
    Returns:
        float : mean l2 loss across batch
    '''

    return torch.nn.functional.mse_loss(src, tgt, reduction='mean')

def smoothness_loss_func(predict, image):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : tensor
            N x 1 x H x W predictions
        image : tensor
            N x 3 x H x W RGB image
        w : tensor
            N x 1 x H x W weights
    Returns:
        tensor : smoothness loss
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y


def sobel_smoothness_loss_func(predict, image, weights, filter_size=[1, 1, 7, 7]):
    '''
    Computes the local smoothness loss using sobel filter

    Arg(s):
        predict : tensor
            N x 1 x H x W predictions
        image : tensor
            N x 3 x H x W RGB image
        w : tensor
            N x 1 x H x W weights
    Returns:
        tensor : smoothness loss
    '''

    device = predict.device

    predict = torch.nn.functional.pad(
        predict,
        (filter_size[-1]//2, filter_size[-1]//2, filter_size[-2]//2, filter_size[-2]//2),
        mode='replicate')

    gx, gy = sobel_filter(filter_size)
    gx = gx.to(device)
    gy = gy.to(device)

    predict_dy = torch.nn.functional.conv2d(predict, gy)
    predict_dx = torch.nn.functional.conv2d(predict, gx)

    image = image[:, 0, :, :] * 0.30 + image[:, 1, :, :] * 0.59 + image[:, 2, :, :] * 0.11
    image = torch.unsqueeze(image, 1)

    image = torch.nn.functional.pad(image, (1, 1, 1, 1), mode='replicate')

    gx_i, gy_i = sobel_filter([1, 1, 3, 3])
    gx_i = gx_i.to(device)
    gy_i = gy_i.to(device)

    image_dy = torch.nn.functional.conv2d(image, gy_i)
    image_dx = torch.nn.functional.conv2d(image, gx_i)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights * weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights * weights_y * torch.abs(predict_dy))

    return (smoothness_x + smoothness_y) / float(filter_size[-1] * filter_size[-2])


'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : tensor
            N x C x H x W tensor
    Returns:
        tensor : gradients in y direction
        tensor : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def sobel_filter(filter_size=[1, 1, 3, 3]):
    Gx = torch.ones(filter_size)
    Gy = torch.ones(filter_size)

    Gx[:, :, :, filter_size[-1] // 2] = 0
    Gx[:, :, (filter_size[-2] // 2), filter_size[-1] // 2 - 1] = 2
    Gx[:, :, (filter_size[-2] // 2), filter_size[-1] // 2 + 1] = 2
    Gx[:, :, :, filter_size[-1] // 2:] = -1*Gx[:, :, :, filter_size[-1] // 2:]

    Gy[:, :, filter_size[-2] // 2, :] = 0
    Gy[:, :, filter_size[-2] // 2 - 1, filter_size[-1] // 2] = 2
    Gy[:, :, filter_size[-2] // 2 + 1, filter_size[-1] // 2] = 2
    Gy[:, :, filter_size[-2] // 2+1:, :] = -1*Gy[:, :, filter_size[-2] // 2+1:, :]

    return Gx, Gy

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.grad_loss_fun = Sobel().cuda()
        self.ssim_loss_fun = SSIM().cuda()
        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)*256.0
            target = target.repeat(1, 3, 1, 1)*256.0
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                # if i == 0:
                #     loss += torch.nn.functional.l1_loss(x, y) + self.grad_loss_fun(x, y) + ((1 - self.ssim_loss_fun(x, y)).mean())*0.5 
                # else:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss