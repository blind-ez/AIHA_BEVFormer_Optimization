import math

import torch
import torch.nn as nn

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.utils import build_from_cfg

from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


@ATTENTION.register_module()
class HeatmapSpatialCrossAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 deformable_attention,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims

        self.deformable_attention = build_from_cfg(deformable_attention, ATTENTION)

        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                value,              # (B, N, ΣHW, C)
                ref_pixel,          # (B, N, P, 2)
                cam_mask,           # (B, N, P)
                spatial_shapes,     # (S, 2)
                level_start_index,  # (S,)
                ref_3d,             # (P, 3)
                **kwargs):
        B, N, _, C = value.shape
        P, _ = ref_3d.shape

        idx_lists = list()
        for batch in range(B):
            idx_list_per_batch = list()
            for cam_mask_per_cam in cam_mask[batch]:
                visible_idxs_per_cam = cam_mask_per_cam.nonzero().squeeze()
                idx_list_per_batch.append(visible_idxs_per_cam)
            idx_lists.append(idx_list_per_batch)
        max_len = max([len(visible_idxs_per_cam) for idx_list_per_batch in idx_lists for visible_idxs_per_cam in idx_list_per_batch])

        ref_pixel_rebatch = ref_pixel.new_zeros(B, N, max_len, 2)  # (B, N, L, 2)
        for batch in range(B):
            idx_list_per_batch = idx_lists[batch]
            for cam, ref_pixel_per_cam in enumerate(ref_pixel[batch]):
                visible_idxs_per_cam = idx_list_per_batch[cam]
                ref_pixel_rebatch[batch, cam, :len(visible_idxs_per_cam)] = ref_pixel_per_cam[visible_idxs_per_cam]

        output = self.deformable_attention(
            value=value.flatten(0, 1),                  # (B*N, ΣHW, C)
            ref_pixel=ref_pixel_rebatch.flatten(0, 1),  # (B*N, L, 2)
            spatial_shapes=spatial_shapes,              # (S, 2)
            level_start_index=level_start_index,        # (S,)
            **kwargs
        )                                               # (B*N, L, C)
        output = output.reshape(B, N, max_len, C)  # (B, N, L, C)

        slots = output.new_zeros([B, P, C])  # (B, P, C)
        for batch in range(B):
            idx_list_per_batch = idx_lists[batch]
            for cam, visible_idxs_per_cam in enumerate(idx_list_per_batch):
                slots[batch, visible_idxs_per_cam] += output[batch, cam, :len(visible_idxs_per_cam)]

        count = cam_mask.sum(1)              # (B, P)
        count = torch.clamp(count, min=1.0)  # (B, P)
        count = count.unsqueeze(-1)          # (B, P, 1)
        slots = slots / count                # (B, P, C)

        slots = self.output_proj(slots)  # (B, P, C)

        out = self.dropout(slots) + self.position_encoder(ref_3d)  # (B, P, C)

        return out


@ATTENTION.register_module()
class HeatmapMultiScaleDeformableAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_scales,
                 num_points,
                 im2col_step=64,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims

        self.num_heads = num_heads
        self.num_scales = num_scales
        self.num_points = num_points

        self.im2col_step = im2col_step

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        self.register_buffer("sampling_offsets", torch.zeros([self.num_heads, self.num_scales, self.num_points, 2]))
        self.register_buffer("attention_weights", torch.full([self.num_heads, self.num_scales, self.num_points], 1.0 / (self.num_scales * self.num_points)))
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)

    def init_weights(self):
        thetas = torch.arange(self.num_heads) * ((2.0 * math.pi) / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(1, keepdim=True).values
        grid_init = grid_init[:, None, None, :].repeat(1, self.num_scales, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.copy_(grid_init)

        xavier_init(self.value_proj, distribution='uniform', bias=0.)

    def forward(self,
                value,              # (B, V, C)
                ref_pixel,          # (B, L, 2)
                spatial_shapes,     # (S, 2)
                level_start_index,  # (S,)
                **kwargs):
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)                 # (S, 2)
        sampling_offsets = self.sampling_offsets / offset_normalizer[None, :, None, :]                        # (h, S, R, 2)
        sampling_locations = ref_pixel[:, :, None, None, None, :] + sampling_offsets[None, None, :, :, :, :]  # (B, L, h, S, R, 2)

        attention_weights = self.attention_weights                                                        # (h, S, R)
        attention_weights = attention_weights[None, None, :, :, :].repeat(*ref_pixel.shape[:2], 1, 1, 1)  # (B, L, h, S, R)

        value = self.value_proj(value)                                # (B, V, C)
        value = value.reshape(*value.shape[:-1], self.num_heads, -1)  # (B, V, h, c)

        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        output = MultiScaleDeformableAttnFunction.apply(
            value,               # (B, V, h, c)
            spatial_shapes,      # (S, 2)
            level_start_index,   # (S,)
            sampling_locations,  # (B, L, h, S, R, 2)
            attention_weights,   # (B, L, h, S, R)
            self.im2col_step     # 64
        ) 

        return output  # (B, L, C)
