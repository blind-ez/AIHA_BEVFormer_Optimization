# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_query', 'bev_pos', 'prev_bev'))
    def preprocess_for_encoder(self,
                               mlvl_feats, # (1, 6, 256, H/8, W/8), (1, 6, 256, H/16, W/16), (1, 6, 256, H/32, W/32), (1, 6, 256, H/64, W/64)
                               bev_query, # (1, 200*200, 256)
                               bev_pos, # (1, 200*200, 256)
                               prev_bev, # (1, 200*200, 256)
                               bev_h, # 200
                               bev_w, # 200
                               grid_length, # [0.512, 0.512]
                               img_meta,
                               **kwargs):

        # get shift
        delta_x = np.array([img_meta['can_bus'][0]])
        delta_y = np.array([img_meta['can_bus'][1]])
        ego_angle = np.array(img_meta['can_bus'][-2] * (180 / np.pi))
        grid_length_x = grid_length[1]
        grid_length_y = grid_length[0]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
        bev_angle = ego_angle - translation_angle
        shift_x = translation_length * np.sin(bev_angle * (np.pi / 180)) / grid_length_x / bev_w
        shift_y = translation_length * np.cos(bev_angle * (np.pi / 180)) / grid_length_y / bev_h
        shift = torch.tensor([shift_x, shift_y], device=bev_query.device, dtype=bev_query.dtype).permute(1, 0)

        # align previous predictions
        if kwargs['apply_occ_mask']:
            prev_preds = kwargs['prev_bbox_preds']
            prev_coords = prev_preds[:, :2]
            prev_velocity = prev_preds[:, -2:]
            occ_reference_coords = prev_coords + prev_velocity * 0.5
            rotation_angle = img_meta['can_bus'][-1] * (np.pi / 180)
            rotation_matrix = torch.tensor([[np.cos(-rotation_angle), -np.sin(-rotation_angle)], [np.sin(-rotation_angle), np.cos(-rotation_angle)]], device=prev_preds.device, dtype=torch.float32)
            occ_reference_coords = occ_reference_coords @ rotation_matrix.T
            occ_reference_coords = occ_reference_coords - shift * 200 * 0.512
            kwargs['occ_reference_coords'].clear()
            kwargs['occ_reference_coords'].append(occ_reference_coords)

        # integrate can_bus info into bev_query
        can_bus = torch.tensor([img_meta['can_bus']], device=bev_query.device, dtype=bev_query.dtype) # (1, 18)
        can_bus = self.can_bus_mlp(can_bus) # (1, 256)
        bev_query = bev_query + can_bus[None, :, :]

        # integrate camera info and level info into mlvl_feats
        updated_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            feat = feat.squeeze() # (6, 256, h, w)
            feat = feat.flatten(2) # (6, 256, h*w)
            feat = feat.permute(0, 2, 1) # (6, h*w, 256)
            feat = feat + self.cams_embeds[:, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, lvl:lvl+1, :].to(feat.dtype)
            updated_feats.append(feat)
        feats_flatten = torch.cat(updated_feats, 1) # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)

        # store the spatial dimensions for each level feature map and computes the cumulative starting index for each level
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            _, _, _, h, w = feat.shape
            spatial_shapes.append((h, w))
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device) # [[116, 200], [58, 100], [29, 50], [15, 25]]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) # [0, 23200, 29000, 30450]

        # rotate prev_bev based on the current timestamp and stack it with the current bev_query
        if prev_bev is not None:
            rotation_angle = img_meta['can_bus'][-1]
            tmp_prev_bev = prev_bev.squeeze().reshape(bev_h, bev_w, -1).permute(2, 0, 1) # (256, 200, 200)
            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center) # (256, 200, 200)
            prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(1, bev_h * bev_w, -1) # (1, 200*200, 256)
            stacked_bev = torch.stack([prev_bev, bev_query], 1) # (1, 2, 200*200, 256)
            stacked_bev = stacked_bev.reshape(2, bev_h * bev_w, -1) # (2, 200*200, 256)
        else:
            stacked_bev = prev_bev # None
            
        return shift, bev_query, feats_flatten, spatial_shapes, level_start_index, stacked_bev

    @auto_fp16(apply_to=('mlvl_feats', 'bev_query', 'bev_pos', 'prev_bev'))
    def get_bev_features(self,
                         mlvl_feats, # (1, 6, 256, H/8, W/8), (1, 6, 256, H/16, W/16), (1, 6, 256, H/32, W/32), (1, 6, 256, H/64, W/64)
                         bev_query, # (1, 200*200, 256)
                         bev_pos, # (1, 200*200, 256)
                         prev_bev, # (1, 200*200, 256)
                         bev_h, # 200
                         bev_w, # 200
                         grid_length, # [0.512, 0.512]
                         img_meta,
                         **kwargs):

        shift, bev_query, feats_flatten, spatial_shapes, level_start_index, stacked_bev = self.preprocess_for_encoder(mlvl_feats, # (1, 6, 256, H/8, W/8), (1, 6, 256, H/16, W/16), (1, 6, 256, H/32, W/32), (1, 6, 256, H/64, W/64)
                                                                                                                      bev_query, # (1, 200*200, 256)
                                                                                                                      bev_pos, # (1, 200*200, 256)
                                                                                                                      prev_bev, # (1, 200*200, 256)
                                                                                                                      bev_h, # 200
                                                                                                                      bev_w, # 200
                                                                                                                      grid_length, # [0.512, 0.512]
                                                                                                                      img_meta,
                                                                                                                      **kwargs)

        bev_query = self.encoder(bev_query=bev_query, # (1, 200*200, 256)
                                 value=feats_flatten, # (6, (H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)+(H/64)(W/64), 256)
                                 bev_pos=bev_pos, # (1, 200*200, 256)
                                 stacked_bev=stacked_bev, # (2, 200*200, 256)
                                 bev_h=bev_h, # 200
                                 bev_w=bev_w, # 200
                                 spatial_shapes=spatial_shapes, # [[116, 200], [58, 100], [29, 50], [15, 25]]
                                 level_start_index=level_start_index, # [0, 23200, 29000, 30450]
                                 shift=shift,
                                 img_meta=img_meta,
                                 **kwargs)

        return bev_query

    @auto_fp16(apply_to=('mlvl_feats', 'bev_query', 'bev_pos', 'object_query', 'object_pos', 'prev_bev'))
    def forward(self,
                mlvl_feats, # (1, 6, 256, H/8, W/8), (1, 6, 256, H/16, W/16), (1, 6, 256, H/32, W/32), (1, 6, 256, H/64, W/64)
                bev_query, # (1, 200*200, 256)
                bev_pos, # (1, 200*200, 256)
                object_query, # (1, 900, 256)
                object_pos, # (1, 900, 256)
                prev_bev, # (1, 200*200, 256)
                bev_h, # 200
                bev_w, # 200
                grid_length, # [0.512, 0.512]
                pc_range,
                img_meta,
                reg_branches,
                cls_branches,
                **kwargs):

        bev_query = self.get_bev_features(mlvl_feats=mlvl_feats, # (1, 6, 256, H/8, W/8), (1, 6, 256, H/16, W/16), (1, 6, 256, H/32, W/32), (1, 6, 256, H/64, W/64)
                                          bev_query=bev_query, # (1, 200*200, 256)
                                          bev_pos=bev_pos, # (1, 200*200, 256)
                                          prev_bev=prev_bev, # (1, 200*200, 256)
                                          bev_h=bev_h, # 200
                                          bev_w=bev_w, # 200
                                          grid_length=grid_length, # [0.512, 0.512]
                                          pc_range=pc_range,
                                          img_meta=img_meta,
                                          **kwargs)

        reference_points = self.reference_points(object_pos) # (1, 900, 3)
        reference_points = reference_points.sigmoid() # (1, 900, 3)

        outputs_classes, outputs_bboxes = self.decoder(query=object_query, # (1, 900, 256)
                                                       value=bev_query, # (1, 200*200, 256)
                                                       query_pos=object_pos, # (1, 900, 256)
                                                       reference_points=reference_points, # (1, 900, 3)
                                                       reg_branches=reg_branches,
                                                       cls_branches=cls_branches,
                                                       pc_range=pc_range,
                                                       spatial_shapes=torch.tensor([[bev_h, bev_w]], device=object_query.device), # [[200, 200]]
                                                       level_start_index=torch.tensor([0], device=object_query.device)) # [0]

        if kwargs['log']:
            bbox_preds = outputs_bboxes[-1, 0]
            cls_scores = outputs_classes[-1, 0]
            cls_scores = cls_scores.max(1).values.sigmoid()

            bbox_preds = bbox_preds[cls_scores > 0.01]
            cls_scores = cls_scores[cls_scores > 0.01]

            kwargs['logger']['output']['bbox_preds'].append(bbox_preds)
            kwargs['logger']['output']['cls_scores'].append(cls_scores)

        return bev_query, outputs_classes, outputs_bboxes