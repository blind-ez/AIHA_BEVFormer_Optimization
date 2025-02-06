# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
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
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
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
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
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
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query, # (1, 200*200, 256)
                value, # (2, 200*200, 256)
                query_pos, # (1, 200*200, 256)
                spatial_shapes, # [[200, 200]]
                level_start_index, # [0]
                ref_2d_hybrid, # (2, 200*200, 1, 2)
                occ_mask,
                **kwargs):

        if occ_mask is not None:
            if kwargs['log']:
                kwargs['sample_logger']['num_queries']['self_attn'] = len(occ_mask)

            _, num_bev_cells, _ = query.shape # 200*200
            num_activated_queries = len(occ_mask)

            if value is None:
                value = query.repeat(2, 1, 1) # (2, 200*200, 256)

            query = query[:, occ_mask, :] # (1, num_activated_queries, 256)
            query_pos = query_pos[:, occ_mask, :] # (1, num_activated_queries, 256)

            identity = query # (1, num_activated_queries, 256)

            query = query + query_pos # (1, num_activated_queries, 256)
            query = torch.cat([value[:1][:, occ_mask, :], query], -1) # (1, num_activated_queries, 256+256)

            sampling_offsets = self.sampling_offsets(query) # (1, num_activated_queries, 128)
            sampling_offsets = sampling_offsets.view(1, num_activated_queries, self.num_heads, 2, 1, self.num_points, 2) # (1, num_activated_queries, 8, 2, 1, 4, 2)
            sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(2, num_activated_queries, self.num_heads, 1, self.num_points, 2) # (2, num_activated_queries, 8, 1, 4, 2)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # [[200, 200]]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :] # (2, num_activated_queries, 8, 1, 4, 2)
            sampling_locations = ref_2d_hybrid[:, occ_mask, :, :][:, :, None, :, None, :] + sampling_offsets # (2, num_activated_queries, 8, 1, 4, 2)

            attention_weights = self.attention_weights(query) # (1, num_activated_queries, 64)
            attention_weights = attention_weights.view(1, num_activated_queries, self.num_heads, 2, 1, self.num_points) # (1, num_activated_queries, 8, 2, 1, 4)
            attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(2, num_activated_queries, self.num_heads, 1, self.num_points).contiguous() # (2, num_activated_queries, 8, 1, 4)
            attention_weights = attention_weights.softmax(-1) # (2, num_activated_queries, 8, 1, 4)

            if kwargs['prune_values']:
                list1 = []
                list2 = []
                for lvl in range(2):
                    tmp = sampling_locations[lvl].reshape(-1, 2)
                    tmp_mask = ((tmp>0.0) & (tmp<1.0)).all(-1)
                    tmp = tmp[tmp_mask]
                    tmp = (tmp * (200-1)).to(torch.int64)
                    tmp = torch.unique(tmp, dim=0)
                    offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=tmp.device)
                    tmp = tmp[:, None, :] + offsets[None, :, :]
                    tmp = torch.unique(tmp.view(-1, 2), dim=0)
                    tmp_flattened = (tmp[:, 1] * 200) + tmp[:, 0]
                    lvl_index = torch.zeros_like(tmp_flattened) + lvl
                    list1.append(tmp_flattened)
                    list2.append(lvl_index)
                aa = torch.cat(list1)
                bb = torch.cat(list2)

                valid_value = self.value_proj(value[bb, aa, :])
                if kwargs['log']:
                    kwargs['sample_logger']['num_values']['self_attn'].append(len(aa))
                value_buffer = torch.zeros_like(value) # (2, 200*200, 256)
                value_buffer[bb, aa, :] = valid_value
                value = value_buffer.reshape(2, num_bev_cells, self.num_heads, -1) # (2, 200*200, 8, 32)
            else:
                value = self.value_proj(value) # (2, 200*200, 256)
                if kwargs['log']:
                    kwargs['sample_logger']['num_values']['self_attn'].append(2*40000)
                value = value.reshape(2, num_bev_cells, self.num_heads, -1) # (2, 200*200, 8, 32)

            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(value, # (2, 200*200, 8, 32)
                                                            spatial_shapes, # [[200, 200]]
                                                            level_start_index, # [0]
                                                            sampling_locations.contiguous(), # (2, num_activated_queries, 8, 1, 4, 2)
                                                            attention_weights, # (2, num_activated_queries, 8, 1, 4)
                                                            self.im2col_step) # 64

            # fuse history value and current value
            output = output.permute(1, 2, 0) # (num_activated_queries, 256, 2)
            output = output.mean(-1) # (num_activated_queries, 256)
            output = self.output_proj(output.unsqueeze(0)) # (1, num_activated_queries, 256)

            return self.dropout(output) + identity

        else:
            if kwargs['log']:
                kwargs['sample_logger']['num_queries']['self_attn'] = 40000

            _, num_bev_cells, _ = query.shape # 200*200

            identity = query # (1, 200*200, 256)

            if value is None:
                value = query.repeat(2, 1, 1) # (2, 200*200, 256)

            query = query + query_pos # (1, 200*200, 256)
            query = torch.cat([value[:1], query], -1) # (1, 200*200, 256+256)

            sampling_offsets = self.sampling_offsets(query) # (1, 200*200, 128)
            sampling_offsets = sampling_offsets.view(1, num_bev_cells, self.num_heads, 2, 1, self.num_points, 2) # (1, 200*200, 8, 2, 1, 4, 2)
            sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(2, num_bev_cells, self.num_heads, 1, self.num_points, 2) # (2, 200*200, 8, 1, 4, 2)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # [[200, 200]]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :] # (2, 200*200, 8, 1, 4, 2)
            sampling_locations = ref_2d_hybrid[:, :, None, :, None, :] + sampling_offsets # (2, 200*200, 8, 1, 4, 2)

            attention_weights = self.attention_weights(query) # (1, 200*200, 64)
            attention_weights = attention_weights.view(1, num_bev_cells, self.num_heads, 2, 1, self.num_points) # (1, 200*200, 8, 2, 1, 4)
            attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(2, num_bev_cells, self.num_heads, 1, self.num_points).contiguous() # (2, 200*200, 8, 1, 4)
            attention_weights = attention_weights.softmax(-1) # (2, 200*200, 8, 1, 4)

            value = self.value_proj(value) # (2, 200*200, 256)
            if kwargs['log']:
                kwargs['sample_logger']['num_values']['self_attn'].append(2*40000)
            value = value.reshape(2, num_bev_cells, self.num_heads, -1) # (2, 200*200, 8, 32)

            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(value, # (2, 200*200, 8, 32)
                                                            spatial_shapes, # [[200, 200]]
                                                            level_start_index, # [0]
                                                            sampling_locations, # (2, 200*200, 8, 1, 4, 2)
                                                            attention_weights, # (2, 200*200, 8, 1, 4)
                                                            self.im2col_step) # 64

            # fuse history value and current value
            output = output.permute(1, 2, 0) # (200*200, 256, 2)
            output = output.mean(-1) # (200*200, 256)
            output = self.output_proj(output.unsqueeze(0)) # (1, 200*200, 256)

            return self.dropout(output) + identity