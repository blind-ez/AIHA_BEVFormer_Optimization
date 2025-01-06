# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'value', 'query_pos', 'ref_3d_projected_to_each_cam'))
    def forward(self,
                query, # (1, 200*200, 256)
                value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                level_start_index, # [0, 23200, 29000, 30450]
                ref_3d_projected_to_each_cam, # (6, 1, 200*200, 4, 2)
                cam_mask): # (6, 1, 200*200, 4)

        _, _, _, num_points_in_pillar, _ = ref_3d_projected_to_each_cam.shape # 4

        identity = query # (1, 200*200, 256)

        indices = []
        for mask_per_cam in cam_mask:
            valid_cell_index_per_cam = mask_per_cam[0].sum(-1).nonzero().squeeze(-1)
            indices.append(valid_cell_index_per_cam)
        max_num = max([len(each) for each in indices])

        # each camera only interacts with its corresponding BEV queries
        # this step can greatly save GPU memory
        query_rebatch = query.new_zeros([6, max_num, self.embed_dims]) # (6, max_num, 256)
        ref_3d_projected_to_each_cam_rebatch = ref_3d_projected_to_each_cam.new_zeros([6, max_num, num_points_in_pillar, 2]) # (6, max_num, 4, 2)
        for i, ref_per_cam in enumerate(ref_3d_projected_to_each_cam):   
            valid_cell_index_per_cam = indices[i]
            query_rebatch[i, :len(valid_cell_index_per_cam)] = query[:, valid_cell_index_per_cam, :]
            ref_3d_projected_to_each_cam_rebatch[i, :len(valid_cell_index_per_cam)] = ref_per_cam[:, valid_cell_index_per_cam, :, :]

        output = self.deformable_attention(query=query_rebatch, # (6, max_num, 256)
                                           value=value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                                           ref_3d_projected_to_each_cam=ref_3d_projected_to_each_cam_rebatch, # (6, max_num, 4, 2)
                                           spatial_shapes=spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                                           level_start_index=level_start_index) # [0, 23200, 29000, 30450]

        # assign attention results for each camera
        slots = torch.zeros_like(query) # (1, 200*200, 256)
        for i, valid_cell_index_per_cam in enumerate(indices):
            slots[:, valid_cell_index_per_cam, :] += output[i, :len(valid_cell_index_per_cam)]

        # calculate the average for bev cells which correspond to more than one camera
        count = cam_mask.sum(-1) > 0 # (6, 1, 200*200)
        count = count.permute(1, 2, 0).sum(-1) # (1, 200*200)
        count = torch.clamp(count, min=1.0) # (1, 200*200)
        slots = slots / count[..., None] # (1, 200*200, 256)

        slots = self.output_proj(slots) # (1, 200*200, 256)

        return self.dropout(slots) + identity


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query, # (6, max_num, 256)
                value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                ref_3d_projected_to_each_cam, # (6, max_num, 4, 2)
                spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                level_start_index): # [0, 23200, 29000, 30450]

        _, num_query, num_points_in_pillar, _ = ref_3d_projected_to_each_cam.shape # max_num, 4

        sampling_offsets = self.sampling_offsets(query) # (6, max_num, 512)
        sampling_offsets = sampling_offsets.view(6, num_query, self.num_heads, self.num_levels, (self.num_points//num_points_in_pillar), num_points_in_pillar, 2) # (6, max_num, 8, 4, 2, 4, 2)
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # [[200, 116], [100, 58], [50, 29], [25, 15]]
        sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, None, :] # (6, max_num, 8, 4, 2, 4, 2)
        sampling_locations = ref_3d_projected_to_each_cam[:, :, None, None, None, :, :] + sampling_offsets # (6, max_num, 8, 4, 2, 4, 2)
        sampling_locations = sampling_locations.view(6, num_query, self.num_heads, self.num_levels, self.num_points, 2) # (6, max_num, 8, 4, 8, 2)

        attention_weights = self.attention_weights(query) # (6, max_num, 256)
        attention_weights = attention_weights.view(6, num_query, self.num_heads, self.num_levels * self.num_points) # (6, max_num, 8, 32)
        attention_weights = attention_weights.softmax(-1) # (6, max_num, 8, 32)
        attention_weights = attention_weights.view(6, num_query, self.num_heads, self.num_levels, self.num_points) # (6, max_num, 8, 4, 8)

        _, num_value, _ = value.shape # (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64)
        value = self.value_proj(value) # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
        value = value.view(6, num_value, self.num_heads, -1) # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 8, 32)

        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        output = MultiScaleDeformableAttnFunction.apply(value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 8, 32)
                                                        spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                                                        level_start_index, # [0, 23200, 29000, 30450]
                                                        sampling_locations, # (6, max_num, 8, 4, 8, 2)
                                                        attention_weights, # (6, max_num, 8, 4, 8)
                                                        self.im2col_step) # 64

        return output # (6, max_num, 256)