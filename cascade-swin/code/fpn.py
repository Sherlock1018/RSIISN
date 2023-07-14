# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch
from ..builder import NECKS

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=256, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CSAttention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7):
        super(CSAttention, self).__init__()


        self.catten = ChannelAttention()
        self.satten = SpatialAttention()
        self.fc1 = nn.Conv2d(in_channel, out_channel, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        ca = self.catten(x) * x
        sa = self.satten(ca) * x
        return sa

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, padding, groups, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0, groups=1, dilation=dilation, bias=False),
            nn.GroupNorm(8,out_channels//4),
           
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=kernel, stride=1, padding=padding, groups=out_channels//4, dilation=dilation, bias=False),
            nn.GroupNorm(8,out_channels//4),
           
            nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0, groups=1, dilation=dilation, bias=False),
            nn.GroupNorm(32,out_channels),
            #nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU()
             )
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
            #nn.ReLU()
            
        self.conv2 = ASPPConv(in_channels, out_channels, kernel=3, padding=1, groups=4, dilation=1)
        self.conv3 = ASPPConv(in_channels, out_channels, kernel=5, padding=2, groups=4, dilation=1)
        self.conv4 = ASPPConv(in_channels, out_channels, kernel=7, padding=3, groups=4, dilation=1)
        self.conv_1_1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_1_2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_1_3 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        #self.conv_1_4 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1_1 = nn.GroupNorm(32,out_channels)
        self.bn1_2 = nn.GroupNorm(32,out_channels)
        self.bn1_3 = nn.GroupNorm(32,out_channels)
        #self.bn1_4 = nn.BatchNorm2d(out_channels)
        #modules.append(ASPPPooling(in_channels, out_channels))
        #self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(128,out_channels),
            #nn.ReLU(),
            #nn.Dropout(0.5)
            )
    def forward(self, x):

        x_p_1 = self.bn1(self.conv1(x))
        x_l_1 = self.bn1_1(self.conv_1_1(x_p_1))
        x_p_2 = self.conv2(x + x_l_1)
        x_l_2 = self.bn1_2(self.conv_1_2(x_p_2))
        x_p_3 = self.conv3(x + x_l_2)
        x_l_3 = self.bn1_3(self.conv_1_3(x_p_3))
        x_p_4 = self.conv4(x + x_l_3)

        res = [x_p_1, x_p_2, x_p_3, x_p_4]
        res = torch.cat(res, dim=1)
        return self.project(res)

@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            #l_conv = ConvModule(
            #    in_channels[i],
            #    out_channels,
            #    1,
            #    conv_cfg=conv_cfg,
            #    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #    act_cfg=act_cfg,
            #    inplace=False)
            l_conv = CSAttention(in_channels[i], out_channels)
            fpn_conv = ASPP(256, [1, 1, 1])
            #fpn_conv = ConvModule(
            #    out_channels,
            #    out_channels,
            #    3,
            #    padding=1,
            #    conv_cfg=conv_cfg,
            #    norm_cfg=norm_cfg,
            #    act_cfg=act_cfg,
            #    inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        #outs = [
        #    self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        #]

        #for i in range(used_backbone_levels):
        level_0 = self.fpn_convs[0](laterals[0])
        level_1 = self.fpn_convs[1](laterals[1]+F.interpolate(level_0, scale_factor=0.5))
        level_2 = self.fpn_convs[2](laterals[2]+F.interpolate(level_1, scale_factor=0.5))
        level_3 = self.fpn_convs[3](laterals[3]+F.interpolate(level_2, scale_factor=0.5))
        outs = [level_0, level_1, level_2, level_3]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
