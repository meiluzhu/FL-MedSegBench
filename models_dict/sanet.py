# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 22:12:20 2026

@author: ZML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#https://github.com/CUMC-Yuan-Lab/GCML/blob/main/SANet/PanSeg/models/sanet1_2_0_v2.py
#from squeeze_and_excitation import ChannelSELayer
BN_EPS = 1e-5
MAX_NUM_FEATURES = 320


from torch.nn.parameter import Parameter
class PopulationNormalization(nn.Module):
    """
    Population Normalization (PN) for federated learning
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(PopulationNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Population statistics as trainable parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Trainable parameters for population statistics
        self.pop_mean = Parameter(torch.zeros(num_features))
        self.pop_gamma = Parameter(torch.ones(num_features))  # gamma^2 (variance)
        
        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.pop_mean)
        nn.init.ones_(self.pop_gamma)
    
    def forward(self, x):
        # x shape: (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=[0, 2, 3] if len(x.shape) == 4 else [0, 2, 3, 4])
            batch_var = x.var(dim=[0, 2, 3] if len(x.shape) == 4 else [0, 2, 3, 4], unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            
            # Use population statistics for normalization
            mean = self.pop_mean
            var = self.pop_gamma
        else:
            # Use population statistics during inference
            mean = self.pop_mean
            var = self.pop_gamma
        
        # Normalize
        if len(x.shape) == 4:  # 2D
            x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        else:  # 3D
            x_norm = (x - mean[None, :, None, None, None]) / torch.sqrt(var[None, :, None, None, None] + self.eps)
        
        # Scale and shift
        if self.affine:
            if len(x.shape) == 4:
                x_norm = x_norm * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            else:
                x_norm = x_norm * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        
        return x_norm
    
    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)


# my implementation of ChannelSELayer
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2, act_layer=nn.ReLU, norm_layer=None, gate_layer=nn.Sigmoid):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.conv_down = nn.Conv3d(num_channels, num_channels_reduced, kernel_size=1, bias=True)
        self.bn = norm_layer(num_channels_reduced) if norm_layer else nn.Identity()

        # self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        # self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = act_layer(inplace=True)
        # self.relu = nn.ELU(inplace=True)
        self.conv_up = nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, bias=True)
        self.sigmoid = gate_layer()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """

        # Average along each channel
        out = self.global_pooling(input_tensor)

        # channel excitation
        out = self.relu(self.bn(self.conv_down(out)))
        out = self.sigmoid(self.conv_up(out))
        out = torch.mul(input_tensor, out)

        return out

# relu2 = nn.ReLU(inplace=True)

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)

class SelectiveScaleLeakyLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, num_scales, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SelectiveScaleLeakyLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

        self.conv_down = nn.Sequential(
            nn.Conv3d(num_channels, num_channels_reduced, kernel_size=1, bias=True),
            #norm(num_channels_reduced, num_channels_per_group=group_norm), # not sure if it's needed
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=False)
        )
        # self.conv_up = nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, bias=False)
        self.conv_up = nn.ModuleList([])
        # self.num_scales = num_scales
        for i in range(num_scales):
            self.conv_up.append(
                nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, bias=True)
            )

        # self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        # self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=False)
        # self.relu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: X, shape = (batch_size, num_scale, num_channels, X, Y, Z)
        :return: output tensor = (batch_size, num_channels, X, Y, Z)
        """

        # Average along each channel
        out = torch.sum(x, dim=1)
        # out = out.mean(-1).mean(-1).mean(-1) # dim = (batch_size x num_channels)
        out = self.global_pooling(out) # batch_size x num_channels x 1 x 1 x 1

        out = self.conv_down(out) # (batch_size x num_channels_reduced x 1 x 1 x 1)
        for i, conv_up in enumerate(self.conv_up):
            # vector = conv_up(out).unsqueeze_(dim=1)
            vector = torch.unsqueeze(conv_up(out), dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        out = self.softmax(attention_vectors) # bz x num_scale x num_channels x 1 x 1 x 1
        out = torch.mul(x, out).sum(dim=1)
        return out

class ConvDropoutInsNormLRelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=(1,1,1), dilation=1, groups=1, padding=1, dropout_p=0.0, norm='in'):
        super(ConvDropoutInsNormLRelu3d, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              dilation=dilation, groups=groups, padding=padding, bias=True)
        if dropout_p > 0.0:
            self.dropout = nn.Dropout3d(p=dropout_p)
        else:
            self.dropout = None
        if norm == 'in':
            self.bn = nn.InstanceNorm3d(num_features=out_channels, eps=1e-5, affine=True)
        if norm == 'pn':
            self.bn = PopulationNormalization(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlockD3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=1, stride=(1,1,1), dropout_p=0.0, se_reduction_ratio=0):
        """
        Implementation of the following ResNet-D:
        He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

        The skip has an avgpool (if needed) followed by 1x1 conv instead of just a strided 1x1 conv
        """

        super(ResidualBlockD3d, self).__init__()
        self.conv1 = ConvDropoutInsNormLRelu3d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size,
                                               padding=padding, stride=stride, dropout_p=dropout_p)
        self.conv2 = ConvDropoutInsNormLRelu3d(in_channels=planes, out_channels=planes, kernel_size=kernel_size,
                                               padding=padding, stride=1, dropout_p=dropout_p)

        has_stride = (isinstance(stride, int) and stride != 1) or (isinstance(stride, tuple) and any([i != 1 for i in stride]))
        need_projection = (inplanes != planes)
        if has_stride or need_projection:
            ops = []
            if has_stride:
                ops.append(nn.AvgPool3d(kernel_size=2, stride=stride, padding=0))
            if need_projection:
                ops.append(ConvDropoutInsNormLRelu3d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1,
                                                     padding=0))
            self.downsample = nn.Sequential(*ops)
        else:
            self.downsample = lambda x:x

        # self.stride = stride
        if se_reduction_ratio > 0:
            self.se = ChannelSELayer(num_channels=planes, reduction_ratio=se_reduction_ratio)
        else:
            self.se = None

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv2(self.conv1(x))
        if self.se is not None:
            out = self.se(out)
        #print('identity:', identity.size())
        #print('out: ', out.size())
        out += identity
        return self.relu(out)



class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, n_blocks, kernel_size=3, padding=1, stride=(2,2,2), dropout_p=0,
                 se_reduction_ratio=0, norm='in'):
        super(StackEncoder, self).__init__()
        assert(n_blocks >= 1)
        # padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=x_channels, out_channels=y_channels, kernel_size=kernel_size,
                                      padding=padding, stride=stride, dropout_p=dropout_p, norm=norm),
            *[ConvDropoutInsNormLRelu3d(in_channels=y_channels, out_channels=y_channels, kernel_size=3,
                                      padding=1, stride=1, dropout_p=dropout_p, norm=norm)
              for i in range(1, n_blocks)]
        )

    def forward(self, x):
        y = self.encode(x)
        return y

class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, n_blocks, kernel_size=(3,3,3), padding=1, stride=(1,1,1),
                 up_mode='upconv', trans_kernel_size=(2,2,2), trans_stride=(2,2,2), dropout_p=0, norm = 'in'):
        super(StackDecoder, self).__init__()
        # padding = (kernel_size - 1) // 2
        assert(n_blocks >= 1)
        if up_mode == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(x_big_channels, x_channels, kernel_size=trans_kernel_size,
                                         stride=trans_stride, bias=True)

        # self.decode = ResidualBlock3d(x_big_channels + x_channels, y_channels, stride=1, downsample=None, group_norm=group_norm)
        self.decode = nn.Sequential(
            # ConvDropoutInsNormLRelu3d(in_channels= 2*x_channels, out_channels=y_channels,
            ConvDropoutInsNormLRelu3d(in_channels=x_channels, out_channels=y_channels,
                                      kernel_size=kernel_size, padding=padding, stride=stride, dropout_p=dropout_p, norm = norm),
            *[ConvDropoutInsNormLRelu3d(in_channels=y_channels, out_channels=y_channels, kernel_size=kernel_size,
                                        padding=padding, stride=1, dropout_p=dropout_p, norm = norm)
              for i in range(1, n_blocks)]
        )

    def forward(self, x_big, x):

        y = self.up(x_big)
        # y = torch.cat([y, x], 1)
        y = y + x
        y = self.decode(y)
        return y

class SANet(nn.Module):
    def __init__(self, in_channels=1, nclasses=1, nf=16, relu='relu',
                 up_mode='upconv', group_norm=0, dropout_p=0, depth=3, padding=True, deep_supervision=False, norm='in'):
        super(SANet, self).__init__()

        self.resdown0 = StackEncoder(x_channels=in_channels, y_channels=min(MAX_NUM_FEATURES, nf),
                                     n_blocks=2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,1,1), norm = norm)
        self.fc01 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            ConvDropoutInsNormLRelu3d(in_channels=min(MAX_NUM_FEATURES, nf),
                                      out_channels=min(MAX_NUM_FEATURES, 2*nf), norm = norm)
        )
        self.fc02 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,4,4), stride=(2,4,4)),
            ConvDropoutInsNormLRelu3d(in_channels=min(MAX_NUM_FEATURES, nf),
                                      out_channels=min(MAX_NUM_FEATURES, 4*nf), norm = norm)
        )
        self.fc03 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4,8,8), stride=(4,8,8)),
            ConvDropoutInsNormLRelu3d(in_channels=min(MAX_NUM_FEATURES, nf),
                                      out_channels=min(MAX_NUM_FEATURES, 8*nf), norm = norm)
        )
        self.ss0 = SelectiveScaleLeakyLayer(num_channels=nf, num_scales=4, reduction_ratio=4)

        self.resdown1 = StackEncoder(x_channels=min(MAX_NUM_FEATURES, nf),   y_channels=min(MAX_NUM_FEATURES, 2*nf),
                                     n_blocks=2, kernel_size=3, stride=(1,2,2), norm = norm) # 2*nf X N/2 X N/2
        self.fc10 = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=2*nf, out_channels=nf, norm = norm),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear',  align_corners=False)
        )
        self.fc12 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvDropoutInsNormLRelu3d(in_channels=min(MAX_NUM_FEATURES,  2 * nf),
                                      out_channels=min(MAX_NUM_FEATURES, 4 * nf), norm = norm)
        )
        self.fc13 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4)),
            ConvDropoutInsNormLRelu3d(in_channels=min(MAX_NUM_FEATURES, 2 * nf),
                                      out_channels=min(MAX_NUM_FEATURES, 8 * nf), norm = norm)
        )
        self.ss1 = SelectiveScaleLeakyLayer(num_channels=2*nf, num_scales=4, reduction_ratio=4)

        self.resdown2 = StackEncoder(x_channels=min(MAX_NUM_FEATURES, 2*nf), y_channels=min(MAX_NUM_FEATURES, 4*nf),
                                     n_blocks=2, kernel_size=3, stride=(2,2,2), norm = norm) # 4*nf X N/4 X N/4
        self.fc20 = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=4 * nf, out_channels=nf, norm = norm),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=False)
        )
        self.fc21 = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=4 * nf, out_channels=2*nf, norm = norm),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        )
        self.fc23 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvDropoutInsNormLRelu3d(in_channels=min(MAX_NUM_FEATURES, 4 * nf),
                                      out_channels=min(MAX_NUM_FEATURES, 8 * nf), norm = norm)
        )
        self.ss2 = SelectiveScaleLeakyLayer(num_channels=4 * nf, num_scales=4, reduction_ratio=4)

        self.resdown3 = StackEncoder(x_channels=min(MAX_NUM_FEATURES, 4*nf), y_channels=min(MAX_NUM_FEATURES, 8*nf),
                                     n_blocks=2, kernel_size=3, stride=(2,2,2), norm = norm) # 8*nf X N/8 X N/8
        self.fc30 = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=8 * nf, out_channels=nf, norm = norm),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=False)
        )
        self.fc31 = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=8 * nf, out_channels=2 * nf, norm = norm),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False)
        )
        self.fc32 = nn.Sequential(
            ConvDropoutInsNormLRelu3d(in_channels=8 * nf, out_channels=4 * nf, norm = norm),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        )
        self.ss3 = SelectiveScaleLeakyLayer(num_channels=8 * nf, num_scales=4, reduction_ratio=4)

        self.resdown4 = StackEncoder(x_channels=min(MAX_NUM_FEATURES, 8*nf), y_channels=min(MAX_NUM_FEATURES, 16*nf),
                                     n_blocks=2, kernel_size=3, stride=(2, 2, 2), norm = norm)  # 8*nf X N/8 X N/8

        self.center_block = StackEncoder(x_channels=min(MAX_NUM_FEATURES, 16*nf), y_channels=min(MAX_NUM_FEATURES, 32*nf),
                                         n_blocks=2, kernel_size=3, stride=(2,2,2), norm = norm) # 16*nf X N/16 X N/16

        self.resup4 = StackDecoder(x_big_channels=min(MAX_NUM_FEATURES, 32*nf), x_channels=min(MAX_NUM_FEATURES, 16*nf),
                                   y_channels=min(MAX_NUM_FEATURES, 16*nf), n_blocks=2, kernel_size=(3,3,3),
                                   padding=(1,1,1), stride=(1,1,1), up_mode=up_mode, norm = norm)  # 4*nf X N/4 X N/4
        self.resup3 = StackDecoder(x_big_channels=min(MAX_NUM_FEATURES, 16*nf), x_channels=min(MAX_NUM_FEATURES, 8*nf),
                                   y_channels=min(MAX_NUM_FEATURES, 8*nf), n_blocks=2, kernel_size=(3,3,3),
                                   padding=(1,1,1), stride=(1,1,1), up_mode=up_mode, norm = norm)  # 4*nf X N/4 X N/4
        self.resup2 = StackDecoder(x_big_channels=min(MAX_NUM_FEATURES, 8*nf), x_channels=min(MAX_NUM_FEATURES, 4*nf),
                                   y_channels=min(MAX_NUM_FEATURES, 4*nf), n_blocks=2, kernel_size=(3,3,3),
                                   padding=(1,1,1), stride=(1,1,1), up_mode=up_mode, norm = norm)  # 4*nf X N/4 X N/4
        self.resup1 = StackDecoder(x_big_channels=min(MAX_NUM_FEATURES, 4*nf), x_channels=min(MAX_NUM_FEATURES, 2*nf),
                                   y_channels=min(MAX_NUM_FEATURES, 2*nf), n_blocks=2, kernel_size=(3,3,3),
                                   padding=(1,1,1), stride=(1,1,1), up_mode=up_mode, norm = norm)  # 4*nf X N/4 X N/4
        self.resup0 = StackDecoder(x_big_channels=min(MAX_NUM_FEATURES, 2*nf), x_channels=min(MAX_NUM_FEATURES, nf),
                                   y_channels=min(MAX_NUM_FEATURES, nf), n_blocks=2, kernel_size=(1,3,3),
                                   padding=(0,1,1), stride=(1,1,1), trans_kernel_size=(1,2,2),
                                   trans_stride=(1,2,2), up_mode=up_mode, norm = norm)  # 4*nf X N/4 X N/4

        self.last = conv1x1x1(in_planes=nf, out_planes=nclasses, stride=1)

        if deep_supervision:
            self.dp4 = conv1x1x1(min(MAX_NUM_FEATURES, 16*nf), nclasses)
            self.dp3 = conv1x1x1(min(MAX_NUM_FEATURES, 8*nf), nclasses)
            self.dp2 = conv1x1x1(min(MAX_NUM_FEATURES, 4*nf), nclasses)
            self.dp1 = conv1x1x1(min(MAX_NUM_FEATURES, 2*nf), nclasses)
        else:
            self.dp4 = None
            self.dp3 = None
            self.dp2 = None
            self.dp1 = None

    def forward(self, x):
        # out = x
        down0 = out = self.resdown0(x)
        down1 = out = self.resdown1(out)

        down2 = out = self.resdown2(out)
        down3 = out = self.resdown3(out)
        down4 = out = self.resdown4(out)

        out = self.center_block(out)


        out = self.resup4(out, down4)

        if self.dp4 is not None:
            dp4_put = self.dp4(out)

        fuse3 = self.ss3(torch.cat([torch.unsqueeze(down3,  dim=1),
                                    torch.unsqueeze(self.fc23(down2), dim=1),
                                    torch.unsqueeze(self.fc13(down1), dim=1),
                                    torch.unsqueeze(self.fc03(down0), dim=1)], dim=1))
        out = self.resup3(out, fuse3)
        # out = self.resup3(out, down3)

        if self.dp3 is not None:
            dp3_out = self.dp3(out)

        fuse2 = self.ss2(torch.cat([torch.unsqueeze(self.fc32(down3), dim=1),
                                    torch.unsqueeze(down2, dim=1),
                                    torch.unsqueeze(self.fc12(down1), dim=1),
                                    torch.unsqueeze(self.fc02(down0), dim=1)], dim=1))
        out = self.resup2(out, fuse2)
        # out = self.resup2(out, down2)

        if self.dp2 is not None:
            dp2_out = self.dp2(out)

        fuse1 = self.ss1(torch.cat([torch.unsqueeze(self.fc31(down3), dim=1),
                                    torch.unsqueeze(self.fc21(down2), dim=1),
                                    torch.unsqueeze(down1, dim=1),
                                    torch.unsqueeze(self.fc01(down0), dim=1)], dim=1))
        out = self.resup1(out, fuse1)
        # out = self.resup1(out, down1)
        # print(out.size())
        # assert (0)

        if self.dp1 is not None:
            dp1_out = self.dp1(out)

        fuse0 = self.ss0(torch.cat([torch.unsqueeze(self.fc30(down3), dim=1),
                                    torch.unsqueeze(self.fc20(down2), dim=1),
                                    torch.unsqueeze(self.fc10(down1), dim=1),
                                    torch.unsqueeze(down0, dim=1)], dim=1))
        out = self.resup0(out, fuse0)
        feature = out

        out = self.last(out)

        if (self.dp4 is None) or (self.training is False):
            return out, feature
        else:
            return [out, dp1_out, dp2_out, dp3_out, dp4_put]

#################################################################3
def main():
    #from torchsummary import summary

    net = SANet(in_channels=1, nclasses=1, nf=16, relu='relu', up_mode='upconv', dropout_p=0, group_norm=1,
              deep_supervision=False)
    n_w = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # n_w = count_parameters(net)
    print('n_w = {0:,}'.format(n_w))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    out = net(torch.randn((1, 1, 32, 128, 128)))
    print(out.shape)
    #summary(net, (1, 64, 128, 128))


if __name__ == '__main__':
    main()