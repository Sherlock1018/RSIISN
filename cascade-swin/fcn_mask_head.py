import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
               
from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit



class OverlapPatchEmbed(nn.Module):              
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=256, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim=64, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.patch_embed = OverlapPatchEmbed(img_size= 14, patch_size=3, stride=1, in_chans=64,
                                              embed_dim=64)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)

        return x

class ResTrans(nn.Module):
    def __init__(self):
        super(ResTrans, self).__init__()
        self.conv_down_t = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_down_t =  nn.GroupNorm(8,64)
        self.conv_up_t = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_up_t =  nn.GroupNorm(32,256)

        self.conv_down_c = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_down_c =  nn.GroupNorm(8,64)
        self.conv_up_c = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_up_c =  nn.GroupNorm(32,256)
        self.trans = Block()
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        B,_,H,W = x.shape
        x_conv = x
        x_conv = self.relu(self.bn_down_c(self.conv_down_c(x_conv)))
        x_conv = self.relu(self.conv0(x_conv)) 
        x_conv = self.bn_up_c(self.conv_up_c(x_conv)) + x
        x = self.relu(self.bn_down_t(self.conv_down_t(x)))
        x = self.trans(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.bn_up_t(self.conv_up_t(x))
        return self.relu(self.conv1(torch.cat([x,x_conv],dim=1)))
        #return self.relu(x_conv)

class Cross_conv(nn.Module):
    def __init__(self):
        super(Cross_conv, self).__init__()
        self.conv2d_1_5 = nn.Conv2d(256, 32, kernel_size=1, padding=0)
        self.conv2d_1_7 = nn.Conv2d(256, 32, kernel_size=1, padding=0)
        self.conv2d_1_9 = nn.Conv2d(256, 32, kernel_size=1, padding=0)
        self.conv2d_1_11 = nn.Conv2d(256, 32, kernel_size=1, padding=0)          
        self.conv2d_5_0 = nn.Conv2d(256, 32, kernel_size=(1,3), padding=(0,int(3/2)))
        self.conv2d_5_1 = nn.Conv2d(256, 32, kernel_size=(3,1), padding=(int(3/2),0))
        self.conv2d_7_0 = nn.Conv2d(256, 32, kernel_size=(1,5), padding=(0,int(5/2)))
        self.conv2d_7_1 = nn.Conv2d(256, 32, kernel_size=(5,1), padding=(int(5/2),0))
        self.conv2d_9_0 = nn.Conv2d(256, 32, kernel_size=(1,7), padding=(0,int(7/2)))
        self.conv2d_9_1 = nn.Conv2d(256, 32, kernel_size=(7,1), padding=(int(7/2),0))
        self.conv2d_11_0 = nn.Conv2d(256, 32, kernel_size=(1,9), padding=(0,int(9/2)))
        self.conv2d_11_1 = nn.Conv2d(256, 32, kernel_size=(9,1), padding=(int(9/2),0))
        self.bn0 = nn.GroupNorm(4,32)
        self.bn1 = nn.GroupNorm(4,32)
        self.bn2 = nn.GroupNorm(4,32)
        self.bn3 = nn.GroupNorm(4,32)
        self.bn4 = nn.GroupNorm(4,32)
        self.bn5 = nn.GroupNorm(4,32)
        self.bn6 = nn.GroupNorm(4,32)
        self.bn7 = nn.GroupNorm(4,32)
        self.bn8 = nn.GroupNorm(4,32)
        self.bn9 = nn.GroupNorm(4,32)
        self.bn10 = nn.GroupNorm(4,32)
        self.bn11 = nn.GroupNorm(4,32)
        self.relu = nn.ReLU()
    def forward(self, x):
        
        #x = torch.flatten(x, start_dim=2)
        x5_3 = self.bn0(self.conv2d_1_5(x))
        x5_0 = self.bn1(self.conv2d_5_0(x))
        x5_1 = self.bn2(self.conv2d_5_1(x))
        x5 = torch.cat((x5_3, (x5_0 + x5_1)), 1)
        x7_3 = self.bn3(self.conv2d_1_7(x))
        x7_0 = self.bn4(self.conv2d_7_0(x))
        x7_1 = self.bn5(self.conv2d_7_1(x))
        x7 = torch.cat((x7_3, (x7_0 + x7_1)), 1)
        x9_3 = self.bn6(self.conv2d_1_9(x))
        x9_0 = self.bn7(self.conv2d_9_0(x))
        x9_1 = self.bn8(self.conv2d_9_1(x))
        x9 = torch.cat((x9_3, (x9_0 + x9_1)), 1)
        x11_3 = self.bn9(self.conv2d_1_11(x))
        x11_0 = self.bn10(self.conv2d_11_0(x))
        x11_1 = self.bn11(self.conv2d_11_1(x))
        x11 = torch.cat((x11_3, (x11_0 + x11_1)), 1)
        x = torch.cat((x5, x7, x9, x11), 1)
        x = self.relu(x)
        return x



@HEADS.register_module()
class FCNMaskHead(BaseModule):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(FCNMaskHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.fp16_enabled = False
        self.loss_mask_bce = build_loss(loss_mask[0])
        self.loss_mask_dice = build_loss(loss_mask[1])
        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.trans1 = ResTrans()
        self.trans2 = ResTrans()
        self.trans3 = ResTrans()
        self.trans4 = ResTrans()
        #self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(256)
        #self.bn2 = nn.BatchNorm2d(256)
        #self.bn3 = nn.BatchNorm2d(256)
        #self.bn4 = nn.BatchNorm2d(256)
        # self.conv1_down = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv2_down = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv3_down = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv4_down = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1_down = nn.BatchNorm2d(128)
        # self.bn2_down = nn.BatchNorm2d(128)
        # self.bn3_down = nn.BatchNorm2d(128)
        # self.bn4_down = nn.BatchNorm2d(128)
        # self.conv1_up = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv2_up = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv3_up = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv4_up = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1_up = nn.BatchNorm2d(256)
        # self.bn2_up = nn.BatchNorm2d(256)
        # self.bn3_up = nn.BatchNorm2d(256)
        # self.bn4_up = nn.BatchNorm2d(256)




    def init_weights(self):
        super(FCNMaskHead, self).init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        B,_,H,W = x.shape
        #for conv in self.convs:
        #    x = conv(x)
        x = self.trans1(x)
        
        x = self.trans2(x)
        
        x = self.trans3(x)
        
        x = self.trans4(x)
        





        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        #from tools.feature_visualization import draw_feature_map
        #draw_feature_map(mask_pred.sigmoid())
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask_bce(mask_pred, mask_targets,
                                           torch.zeros_like(labels)) + self.loss_mask_dice(mask_pred, mask_targets)
            else:
                num_rois = mask_pred.size()[0]
                inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
                mask_pred_slice = mask_pred[inds, labels].squeeze(1)
               
                loss_mask = self.loss_mask_bce(mask_pred, mask_targets, labels) + self.loss_mask_dice(mask_pred_slice, mask_targets)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(float | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        Example:
            >>> import mmcv
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> det_labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = torch.FloatTensor((1, 1))
            >>> rescale = False
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self.get_seg_masks(
            >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
            >>>     scale_factor, rescale
            >>> )
            >>> assert len(encoded_masks) == C
            >>> assert sum(list(map(len, encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        # No need to consider rescale and scale_factor while exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            img_h, img_w = ori_shape[:2]
        else:
            if rescale:
                img_h, img_w = ori_shape[:2]
            else:
                if isinstance(scale_factor, float):
                    img_h = np.round(ori_shape[0] * scale_factor).astype(
                        np.int32)
                    img_w = np.round(ori_shape[1] * scale_factor).astype(
                        np.int32)
                else:
                    w_scale, h_scale = scale_factor[0], scale_factor[1]
                    img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                        np.int32)
                    img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                        np.int32)
                scale_factor = 1.0

            if not isinstance(scale_factor, (float, torch.Tensor)):
                scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes / scale_factor

        # support exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            threshold = rcnn_test_cfg.mask_thr_binary
            if not self.class_agnostic:
                box_inds = torch.arange(mask_pred.shape[0])
                mask_pred = mask_pred[box_inds, labels][:, None]
            masks, _ = _do_paste_mask(
                mask_pred, bboxes, img_h, img_w, skip_empty=False)
            if threshold >= 0:
                masks = (masks >= threshold).to(dtype=torch.bool)
            else:
                # TensorRT backend does not have data type of uint8
                is_trt_backend = os.environ.get(
                    'ONNX_BACKEND') == 'MMCVTensorRT'
                target_dtype = torch.int32 if is_trt_backend else torch.uint8
                masks = (masks * 255).to(dtype=target_dtype)
            return masks

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
