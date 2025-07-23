# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.model.weight_init import kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint

from mmaction.registry import MODELS
from .resnet3d import ResNet3d

import numpy as np
import torch
from torch import nn
from torch.nn import init

class SELayer(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, in_channels=2048, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv3d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm3d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, t, h,w = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准


class DeConvModule(BaseModule):
    """A deconv module that bundles deconv/norm/activation layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input.
        bias (bool): Whether to add a learnable bias to the output.
            Defaults to False.
        with_bn (bool): Whether to add a BN layer. Defaults to True.
        with_relu (bool): Whether to add a ReLU layer. Defaults to True.
    """

    def __init__(self,
                 in_channels: int,  # 输入特征图的通道数
                 out_channels: int, # 卷积操作输出的通道数
                 kernel_size: int,  # 反卷积核的大小
                 stride: Union[int, Tuple[int]] = (1, 1, 1),   # 反卷积的步长，默认为 (1, 1, 1)
                 padding: Union[int, Tuple[int]] = 0,        # 输入两侧填充的大小，默认为 0
                 bias: bool = False,      # 是否在输出中添加可学习的偏置项，默认为 False
                 with_bn: bool = True,   # 是否添加批量归一化（Batch Normalization）层，默认为 True
                 with_relu: bool = True) -> None:   # 是否添加 ReLU 激活函数层，默认为 True
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu
        # 定义一个 3D 反卷积层
        self.conv = nn.ConvTranspose3d(
            in_channels, # 输入特征图的通道数
            out_channels, # 卷积操作输出的通道数
            kernel_size,   # 反卷积核的大小
            stride=stride,  # 反卷积的步长
            padding=padding,   # 输入两侧填充的大小
            bias=bias)       # 是否添加可学习的偏置项
        self.bn = nn.BatchNorm3d(out_channels)  # 定义一个 3D 批量归一化层
        self.relu = nn.ReLU()       # 定义一个 ReLU 激活函数层
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        # x should be a 5-d tensor
        # 检查输入张量 x 是否为 5 维张量，因为 3D 反卷积操作通常处理 5 维输入
        # 5 个维度分别代表：批次大小 (N)、通道数 (C)、时间维度 (T)、高度 (H) 和宽度 (W)
        assert len(x.shape) == 5
        # 解包输入张量 x 的形状，获取批次大小、通道数、时间维度、高度和宽度
        N, C, T, H, W = x.shape
        # 计算反卷积输出的形状
        # 时间维度、高度和宽度会根据反卷积的步长进行扩展
        out_shape = (N, self.out_channels, self.stride[0] * T,
                     self.stride[1] * H, self.stride[2] * W)
        #对输入张量 x 进行 3D 反卷积操作，指定输出形状
        x = self.conv(x, output_size=out_shape)
        # 如果 with_bn 标志为 True，则对反卷积的输出进行批量归一化操作
        if self.with_bn:
            x = self.bn(x)
        # 如果 with_relu 标志为 True，则对经过批量归一化的输出应用 ReLU 激活函数
        if self.with_relu:
            x = self.relu(x)
        return x


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Defaults to False.
        lateral_inv (bool): Whether to use deconv to upscale the time
            dimension of features from another pathway. Defaults to False.
        lateral_norm (bool): Determines whether to enable the lateral norm
            in lateral layers. Defaults to False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Defaults to 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Defaults to 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Defaults to 5.
        lateral_infl (int): The ratio of the inflated channels.
            Defaults to 2.
        lateral_activate (list[int]): Flags for activating the lateral
            connection. Defaults to ``[1, 1, 1, 1]``.
    """
#用于构建 SlowFast 网络中的慢速或快速路径。
    def __init__(self,
                 lateral: bool = False,# 是否启用横向连接
                 lateral_inv: bool = False,#是否使用反卷积来上采样另一个路径特征的时间维度。默认为 False。
                 lateral_norm: bool = False,#是否在横向层中启用归一化。默认为 False。
                 speed_ratio: int = 8,  #快速路径和慢速路径时间维度的比率，对应论文中的 `alpha`。默认为 8。
                 channel_ratio: int = 8, # 快速路径通道数的缩减比例，对应论文中的 `beta`。默认为 8。
                 fusion_kernel: int = 5,#横向融合的卷积核大小。默认为 5。
                 lateral_infl: int = 2,  #膨胀通道的比率。默认为 2。
                 lateral_activate: List[int] = [1, 1, 1, 1],#激活横向连接的标志列表。默认为 [1, 1, 1, 1]。
                 **kwargs) -> None:
        self.lateral = lateral
        self.lateral_inv = lateral_inv
        self.lateral_norm = lateral_norm
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = lateral_activate
        # 计算横向连接的输入通道数
        self._calculate_lateral_inplanes(kwargs)

        super().__init__(**kwargs)
        self.inplanes = self.base_channels   #初始化输入通道数为基础通道数
        if self.lateral and self.lateral_activate[0] == 1:  #如果启用横向连接且第一个阶段的横向连接被激活
            if self.lateral_inv:
                # 使用反卷积模块创建第一个横向连接层
                self.conv1_lateral = DeConvModule(
                    self.inplanes * self.channel_ratio,  # 输入通道数
                    self.inplanes * self.channel_ratio // lateral_infl,  # 输出通道数
                    kernel_size=(fusion_kernel, 1, 1),  # 卷积核大小
                    stride=(self.speed_ratio, 1, 1),    # 步长
                    padding=((fusion_kernel - 1) // 2, 0, 0),   # 填充大小
                    with_bn=True,   # 是否使用批量归一化
                    with_relu=True) # 是否使用 ReLU 激活函数
            else:
                 # 使用卷积模块创建第一个横向连接层
                self.conv1_lateral = ConvModule(
                    self.inplanes // self.channel_ratio,  # 输入通道数
                    self.inplanes * lateral_infl // self.channel_ratio, # 输出通道数
                    kernel_size=(fusion_kernel, 1, 1),   # 卷积核大小
                    stride=(self.speed_ratio, 1, 1),  # 步长
                    padding=((fusion_kernel - 1) // 2, 0, 0),  # 填充大小
                    bias=False,  # 是否使用偏置
                    conv_cfg=self.conv_cfg,  # 卷积配置
                    norm_cfg=self.norm_cfg if self.lateral_norm else None,   # 归一化配置
                    act_cfg=self.act_cfg if self.lateral_norm else None)     # 激活函数配置

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1 \
                    and self.lateral_activate[i + 1]:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                if self.lateral_inv:
                    conv_module = DeConvModule(
                        self.inplanes * self.channel_ratio,
                        self.inplanes * self.channel_ratio // lateral_infl,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        with_bn=True,
                        with_relu=True)
                else:
                    conv_module = ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * lateral_infl // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg if self.lateral_norm else None,
                        act_cfg=self.act_cfg if self.lateral_norm else None)
                setattr(self, lateral_name, conv_module)
                self.lateral_connections.append(lateral_name)

    def _calculate_lateral_inplanes(self, kwargs):
        """Calculate inplanes for lateral connection."""
        depth = kwargs.get('depth', 50)
        expansion = 1 if depth < 50 else 4
        base_channels = kwargs.get('base_channels', 64)
        lateral_inplanes = []
        for i in range(kwargs.get('num_stages', 4)):
            if expansion % 2 == 0:
                planes = base_channels * (2 ** i) * \
                         ((expansion // 2) ** (i > 0))
            else:
                planes = base_channels * (2**i) // (2**(i > 0))
            if self.lateral and self.lateral_activate[i]:
                if self.lateral_inv:
                    lateral_inplane = planes * \
                                      self.channel_ratio // self.lateral_infl
                else:
                    lateral_inplane = planes * \
                                      self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
            lateral_inplanes.append(lateral_inplane)
        self.lateral_inplanes = lateral_inplanes

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the ``lateral_connection`` part should
        not be inflated from 2d weights.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d,
                                              original_conv_name,
                                              inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def _inflate_conv_params(self, conv3d: nn.Module,
                             state_dict_2d: OrderedDict, module_name_2d: str,
                             inflated_param_names: List[str]) -> None:
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = torch.cat(
                (conv2d_weight,
                 torch.zeros(pad_shape).type_as(conv2d_weight).to(
                     conv2d_weight.device)),
                dim=1)

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        # 如果提供了预训练模型的路径，则更新 self.pretrained 属性
        if pretrained:
            self.pretrained = pretrained
        # 调用父类的 init_weights 方法来初始化权重，覆盖 I3D 的权重初始化方法
        # Override the init_weights of i3d
        super().init_weights()
        # 遍历所有横向连接层
        for module_name in self.lateral_connections:
            # 获取当前横向连接层的模块
            layer = getattr(self, module_name)
             # 遍历当前横向连接层模块中的所有子模块
            for m in layer.modules():
                # 如果子模块是 3D 卷积层或 2D 卷积层
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    # 使用 Kaiming 初始化方法对卷积层的权重进行初始化
                    kaiming_init(m)


pathway_cfg = {
    'resnet3d': ResNet3dPathway,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build pathway.

    Args:
        cfg (dict): cfg should contain:
            - type (str): identify backbone type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@MODELS.register_module()
class ResNet3dSlowFast(BaseModule):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str):   预训练模型的文件路径。通过指定该路径，可以加载一个已经训练好的模型权重，用于初始化当前模型。
        resample_rate (int): 输入帧的时间步长采样率。实际采样率由pipeline中SampleFrames的interval参数乘以resample_rate决定,对应论文中的τ参数。例如设为8表示每8*interval帧只处理1帧。默认为8。
        speed_ratio (int): 快慢路径的时间维度比例,对应论文中的a参数。默认为8。
        channel_ratio (int): 快路径的通道数缩减比例,对应论文中的β参数。默认为8
        slow_pathway (dict): Configuration of slow branch. Defaults to
            ``dict(type='resnet3d', depth=50, pretrained=None, lateral=True,
            conv1_kernel=(1, 7, 7), conv1_stride_t=1, pool1_stride_t=1,
            inflate=(0, 0, 1, 1))``.
        fast_pathway (dict): Configuration of fast branch. Defaults to
            ``dict(type='resnet3d', depth=50, pretrained=None, lateral=False,
            base_channels=8, conv1_kernel=(5, 7, 7), conv1_stride_t=1,
            pool1_stride_t=1)``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 resample_rate: int = 8, # 输入帧的时间步长，实际重采样率为流水线中 SampleFrames 的 interval 乘以 resample_rate
                 speed_ratio: int = 8, # 快速路径和慢速路径时间维度的比率，对应论文中的 α
                 channel_ratio: int = 8,# 快速路径通道数的缩减比例，对应论文中的 β
                 slow_pathway: Dict = dict(
                    #slow_pathway：类型为字典（dict），用于配置慢速分支的参数。默认配置使用深度为 50 的 3D ResNet，
                    #不使用预训练模型，开启横向连接，第一层卷积核大小为 (1, 7, 7)，
                    # 第一层卷积在时间维度上的步长为 1，第一层池化在时间维度上的步长为 1，
                    # 并且在特定阶段进行膨胀操作
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway: Dict = dict(
                     #用于配置快速分支的参数。默认配置同样使用深度为 50 的 3D ResNet，
                     # 不使用预训练模型，不开启横向连接，基础通道数为 8，第一层卷积核大小为 (5, 7, 7)，
                     # 第一层卷积在时间维度上的步长为 1，第一层池化在时间维度上的步长为 1
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.resample_rate = resample_rate  # 保存输入帧的重采样率
        self.speed_ratio = speed_ratio      # 保存快速路径和慢速路径时间维度的比率
        self.channel_ratio = channel_ratio  # 保存快速路径通道数的缩减比例

        # 如果慢速分支启用横向连接
        if slow_pathway['lateral']:
            # 将速度比率添加到慢速分支的配置中,参数τ，
            slow_pathway['speed_ratio'] = speed_ratio
            # 将通道比率添加到慢速分支的配置中,参数α
            slow_pathway['channel_ratio'] = channel_ratio
         # 根据配置构x建慢速分支
        self.slow_path = build_pathway(slow_pathway)
         # 根据配置构建慢速分支
        self.fast_path = build_pathway(fast_pathway)
        last_layer_out_channels = 2048
        self.se_slow_path = SELayer(last_layer_out_channels)


        
    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained
        # 检查 pretrained 是否为字符串类型
        if isinstance(self.pretrained, str):
            # 获取当前的日志记录器实例
            logger = MMLogger.get_current_instance()
            # 生成加载模型的提示信息
            msg = f'load model from: {self.pretrained}'
            # 使用日志记录器打印提示信息
            print_log(msg, logger=logger)
            # 直接加载 3D 预训练模型，strict=True 表示严格匹配模型参数
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples
                extracted by the backbone.
        """
        #print(f"Input shape: {x.shape}")        # B,C,T,W,H -> ([16, 3, 32, 256, 256])
        # 对输入数据 x 进行下采样，得到慢速路径的输入
        # 下采样的比例在时间维度上为 1/self.resample_rate，空间维度保持不变
        x_slow = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        
        #print(f"Shape of x_slow after interpolation: {x_slow.shape}")  # B,C,T,W,H -> ([16, 3, 4, 256, 256])    1/8
        # 将下采样后的输入数据传入慢速路径的第一个卷积层
        x_slow = self.slow_path.conv1(x_slow)       #Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        #print(f"Shape of x_slow after conv1: {x_slow.shape}")          # B,C,T,W,H -> ([16, 64, 4, 128, 128]) 
        # 对卷积层的输出进行最大池化操作
        x_slow = self.slow_path.maxpool(x_slow)      #self.slow_path.maxpool.kernel_size = (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        #print(f"Shape of x_slow after maxpool: {x_slow.shape}")        # B,C,T,W,H -> ([16, 64, 4, 64, 64])

        # 对输入数据 x 进行下采样，得到快速路径的输入
        # 下采样的比例在时间维度上为 1/(self.resample_rate // self.speed_ratio)，空间维度保持不变
        x_fast = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0,
                          1.0))
        # 将下采样后的输入数据传入快速路径的第一个卷积层
        #print(f"Shape of x_fast after interpolation: {x_fast.shape}")     # B,C,T,W,H -> ([16, 3, 32, 256, 256])    8/8
        x_fast = self.fast_path.conv1(x_fast)                             #Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        #print(f"Shape of x_fast after conv1: {x_fast.shape}")             # B,C,T,W,H -> ([16, 8, 32, 128, 128])
        # 对卷积层的输出进行最大池化操作
        x_fast = self.fast_path.maxpool(x_fast)                           #self.fast_path.maxpool.kernel_size = (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        #print(f"Shape of x_fast after maxpool: {x_fast.shape}")           # B,C,T,W,H -> ([16, 8, 32, 64, 64])

        # 如果慢速路径启用了横向连接
        if self.slow_path.lateral:
            # 将快速路径的特征图通过慢速路径的第一个横向连接层
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)        # Conv3d(8, 16, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
            #print(f"Shape of x_fast_lateral after conv1_lateral: {x_fast_lateral.shape}")  # B,C,T,W,H         ([16, 8, 32, 64, 64])-> ([16, 16, 4, 64, 64])
            # 在通道维度上拼接慢速路径的特征图和快速路径经过横向连接层后的特征图
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)         
            #print(f"Shape of x_slow after cat with x_fast_lateral: {x_slow.shape}")      # B,C,T,W,H         ([16, 16, 4, 64, 64]) , ([16, 64,4,64, 64]) -> ([16, 80, 4, 64, 64])
        # 遍历慢速路径的每一个残差层
        for i, layer_name in enumerate(self.slow_path.res_layers):                       
            # 获取慢速路径的当前残差层
            res_layer = getattr(self.slow_path, layer_name)                              
            # 将慢速路径的特征图传入当前残差层
            x_slow = res_layer(x_slow)
            #print(f"Shape of x_slow after {layer_name}: {x_slow.shape}")          # B,C,T,W,H -> ([16, 256, 4, 64, 64])        
            # 获取快速路径的对应残差层
            res_layer_fast = getattr(self.fast_path, layer_name)                  #B,C,T,W,H ->  ([16,32,32,64,64])
            # 将快速路径的特征图传入对应残差层
            x_fast = res_layer_fast(x_fast)
            #print(f"Shape of x_fast after {layer_name}: {x_fast.shape}") 
             # 如果不是最后一个残差层且慢速路径启用了横向连接
            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral):
                # No fusion needed in the final stage
                # 最后一个阶段不需要融合操作
                # 获取当前阶段的横向连接层名称
                lateral_name = self.slow_path.lateral_connections[i]
                # 获取当前阶段的横向连接层
                conv_lateral = getattr(self.slow_path, lateral_name)
                # 将快速路径的特征图通过当前阶段的横向连接层
                x_fast_lateral = conv_lateral(x_fast)
                #print(f"Shape of x_fast_lateral after {lateral_name}: {x_fast_lateral.shape}")  #Shape of x_fast_lateral after layer1_lateral: torch.Size([16, 64, 4, 64, 64])
                # 在通道维度上拼接慢速路径的特征图和快速路径经过当前横向连接层后的特征图
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
                #print(f"Shape of x_slow after cat with {lateral_name} output: {x_slow.shape}")  #Shape of x_slow after cat with layer1_lateral output: torch.Size([16, 320, 4, 64, 64])
        # 在慢路径的最后一个残差块后面应用 SE 模块
        x_slow = self.se_slow_path(x_slow)
        #print(f"Shape of x_slow after SE layer: {x_slow.shape}")
        # 将慢速路径和快速路径的最终特征图组合成一个元组
        out = (x_slow, x_fast)
        print(f"Output shapes: Slow - {x_slow.shape}, Fast - {x_fast.shape}")
        return out
#Input shape: torch.Size([16, 3, 32, 256, 256])
#Shape of x_slow after interpolation: torch.Size([16, 3, 4, 256, 256])
#Shape of x_slow after conv1: torch.Size([16, 64, 4, 128, 128])
#Shape of x_slow after maxpool: torch.Size([16, 64, 4, 64, 64])
#Shape of x_fast after interpolation: torch.Size([16, 3, 32, 256, 256])
#Shape of x_fast after conv1: torch.Size([16, 8, 32, 128, 128])
#Shape of x_fast after maxpool: torch.Size([16, 8, 32, 64, 64])
#Shape of x_fast_lateral after conv1_lateral: torch.Size([16, 16, 4, 64, 64])
#Shape of x_slow after cat with x_fast_lateral: torch.Size([16, 80, 4, 64, 64])
#Shape of x_slow after layer1: torch.Size([16, 256, 4, 64, 64])
#Shape of x_fast after layer1: torch.Size([16, 32, 32, 64, 64])
#Shape of x_fast_lateral after layer1_lateral: torch.Size([16, 64, 4, 64, 64])
#Shape of x_slow after cat with layer1_lateral output: torch.Size([16, 320, 4, 64, 64])
#Shape of x_slow after layer2: torch.Size([16, 512, 4, 32, 32])
#Shape of x_fast after layer2: torch.Size([16, 64, 32, 32, 32])
# Shape of x_fast_lateral after layer2_lateral: torch.Size([16, 128, 4, 32, 32])
# Shape of x_slow after cat with layer2_lateral output: torch.Size([16, 640, 4, 32, 32])
# Shape of x_slow after layer3: torch.Size([16, 1024, 4, 16, 16])
# Shape of x_fast after layer3: torch.Size([16, 128, 32, 16, 16])
# Shape of x_fast_lateral after layer3_lateral: torch.Size([16, 256, 4, 16, 16])
# Shape of x_slow after cat with layer3_lateral output: torch.Size([16, 1280, 4, 16, 16])
# Shape of x_slow after layer4: torch.Size([16, 2048, 4, 16, 16])
# Shape of x_fast after layer4: torch.Size([16, 256, 32, 16, 16])