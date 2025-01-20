# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    def get_reference_points(self, bev_h, bev_w, pc_range, shift, img_meta, device, dtype):
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device),
                                      torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device))
        ref_y = ref_y.reshape(-1)[None] / bev_h
        ref_x = ref_x.reshape(-1)[None] / bev_w
        ref_2d = torch.stack((ref_x, ref_y), -1) # (1, 200*200, 2)
        ref_2d = ref_2d.unsqueeze(2) # (1, 200*200, 1, 2)

        ref_2d_shift = ref_2d + shift[:, None, None, :] # (1, 200*200, 1, 2)
        ref_2d_hybrid = torch.stack([ref_2d_shift, ref_2d], 1) # (1, 2, 200*200, 1, 2)
        ref_2d_hybrid = ref_2d_hybrid.reshape(2, bev_h * bev_w, 1, 2) # (2, 200*200, 1, 2)

        # reference points in 3D space, used in spatial cross-attention (SCA)
        Z = pc_range[5] - pc_range[2] # 8.0
        zs = torch.linspace(0.5, Z - 0.5, self.num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(self.num_points_in_pillar, bev_h, bev_w) / Z
        xs = torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device).view(1, 1, bev_w).expand(self.num_points_in_pillar, bev_h, bev_w) / bev_w
        ys = torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device).view(1, bev_h, 1).expand(self.num_points_in_pillar, bev_h, bev_w) / bev_h
        ref_3d = torch.stack((xs, ys, zs), -1) # (4, 200, 200, 3)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # (4, 200*200, 3)
        ref_3d = ref_3d.unsqueeze(0) # (1, 4, 200*200, 3)

        ref_3d_projected_to_each_cam, cam_mask = self.point_sampling(ref_3d, pc_range, img_meta) # (6, 1, 200*200, 4, 2), (6, 1, 200*200, 4)

        return ref_2d_hybrid, ref_3d_projected_to_each_cam, cam_mask

    # This function must use fp32!!!
    @force_fp32(apply_to=('ref_3d', 'img_meta'))
    def point_sampling(self, ref_3d, pc_range, img_meta):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = ref_3d.new_tensor(lidar2img)
        ref_3d = ref_3d.clone()

        ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        ref_3d = torch.cat((ref_3d, torch.ones_like(ref_3d[..., :1])), -1)

        ref_3d = ref_3d.permute(1, 0, 2, 3)
        D, B, num_query = ref_3d.size()[:3]
        num_cam = lidar2img.size(1)

        ref_3d = ref_3d.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        ref_3d_projected_to_each_cam = torch.matmul(lidar2img.to(torch.float32), ref_3d.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        cam_mask = (ref_3d_projected_to_each_cam[..., 2:3] > eps)
        ref_3d_projected_to_each_cam = ref_3d_projected_to_each_cam[..., 0:2] / torch.maximum(ref_3d_projected_to_each_cam[..., 2:3], torch.ones_like(ref_3d_projected_to_each_cam[..., 2:3]) * eps)

        ref_3d_projected_to_each_cam[..., 0] /= img_meta['img_shape'][0][1]
        ref_3d_projected_to_each_cam[..., 1] /= img_meta['img_shape'][0][0]

        cam_mask &= (ref_3d_projected_to_each_cam[..., 1:2] > 0.0) \
                    & (ref_3d_projected_to_each_cam[..., 1:2] < 1.0) \
                    & (ref_3d_projected_to_each_cam[..., 0:1] < 1.0) \
                    & (ref_3d_projected_to_each_cam[..., 0:1] > 0.0)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            cam_mask = torch.nan_to_num(cam_mask)
        else:
            cam_mask = cam_mask.new_tensor(np.nan_to_num(cam_mask.cpu().numpy()))

        ref_3d_projected_to_each_cam = ref_3d_projected_to_each_cam.permute(2, 1, 3, 0, 4)
        cam_mask = cam_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return ref_3d_projected_to_each_cam, cam_mask

    @auto_fp16()
    def forward(self,
                bev_query, # (1, 200*200, 256)
                value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                bev_pos, # (1, 200*200, 256)
                bev_h, # 200
                bev_w, # 200
                pc_range,
                shift,
                img_meta,
                **kwargs):

        ref_2d_hybrid, ref_3d_projected_to_each_cam, cam_mask = self.get_reference_points(bev_h=bev_h,
                                                                                          bev_w=bev_w,
                                                                                          pc_range=pc_range,
                                                                                          shift=shift,
                                                                                          img_meta=img_meta,
                                                                                          device=bev_query.device,
                                                                                          dtype=bev_query.dtype) # (2, 200*200, 1, 2), (1, 4, 200*200, 3), (6, 1, 200*200, 4)

        if kwargs['prev_preds'] is not None:
            n = 25
            bev_size = bev_h

            prev_preds = kwargs['prev_preds']
            normalized_coords = (prev_preds - (-51.2)) / 102.4
            bev_coords = (normalized_coords * bev_size).to(torch.long)

            xs = (torch.arange(n)-(n//2))[:, None].repeat(1, n).to(bev_query.device)
            ys = (torch.arange(n)-(n//2))[None, :].repeat(n, 1).to(bev_query.device)
            offsets = torch.stack((xs, ys), dim=2).view(-1, 2)[None, :, :]

            expanded_coords = bev_coords[:, None, :] + offsets
            unique_coords = torch.unique(expanded_coords.view(-1, 2), dim=0)
            valid_mask = ((unique_coords < bev_size) & (unique_coords >= 0)).all(1)
            valid_coords = unique_coords[valid_mask]

            occ_mask = (valid_coords[:, 1] * bev_size) + valid_coords[:, 0]
            kwargs['occ_mask'] = occ_mask

            kwargs['mask_logger'].append(valid_coords.to('cpu'))
        else:
            kwargs['mask_logger'].append(None)

        for lid, layer in enumerate(self.layers):
            bev_query = layer(query=bev_query, # (1, 200*200, 256)
                              value=value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                              bev_pos=bev_pos, # (1, 200*200, 256)
                              bev_h=bev_h, # 200
                              bev_w=bev_w, # 200
                              ref_2d_hybrid=ref_2d_hybrid, # (2, 200*200, 1, 2)
                              ref_3d_projected_to_each_cam=ref_3d_projected_to_each_cam, # (6, 1, 200*200, 4, 2)
                              cam_mask=cam_mask, # (6, 1, 200*200, 4)
                              **kwargs)

        return bev_query


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query, # (1, 200*200, 256)
                value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                bev_pos, # (1, 200*200, 256)
                stacked_bev, # (2, 200*200, 256)
                bev_h, # 200
                bev_w, # 200
                spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                level_start_index, # [0, 23200, 29000, 30450]
                ref_2d_hybrid, # (2, 200*200, 1, 2)
                ref_3d_projected_to_each_cam, # (6, 1, 200*200, 4, 2)
                cam_mask, # (6, 1, 200*200, 4)
                occ_mask=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0

        for layer in self.operation_order:
            if layer == 'self_attn': # temporal self attention
                query = self.attentions[attn_index](query=query, # (1, 200*200, 256)
                                                    value=stacked_bev, # (2, 200*200, 256)
                                                    query_pos=bev_pos, # (1, 200*200, 256)
                                                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device), # [[200, 200]]
                                                    level_start_index=torch.tensor([0], device=query.device), # [0]
                                                    ref_2d_hybrid=ref_2d_hybrid, # (2, 200*200, 1, 2)
                                                    occ_mask=occ_mask)
                attn_index += 1

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn': # spaital cross attention
                query = self.attentions[attn_index](query=query, # (1, 200*200, 256)
                                                    value=value, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                                                    spatial_shapes=spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                                                    level_start_index=level_start_index, # [0, 23200, 29000, 30450]
                                                    ref_3d_projected_to_each_cam=ref_3d_projected_to_each_cam, # (6, 1, 200*200, 4, 2)
                                                    cam_mask=cam_mask, # (6, 1, 200*200, 4)
                                                    occ_mask=occ_mask)
                attn_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, None)
                ffn_index += 1

        if occ_mask is not None:
            query_zero_padded = query.new_zeros([1, bev_h * bev_w, self.embed_dims]) # (1, 200*200, 256)
            query_zero_padded[:, occ_mask, :] = query
            return query_zero_padded

        return query