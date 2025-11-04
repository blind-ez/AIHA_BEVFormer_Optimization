import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmdet.models import HEADS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@HEADS.register_module()
class HeatmapHead(MVXTwoStageDetector):
    def __init__(self,
                 embed_dims,
                 heatmap_size,
                 heatmap_encoder,
                 heatmap_decoder):
        super().__init__()

        self.embed_dims = embed_dims

        self.Hh, self.Wh = heatmap_size

        self.heatmap_encoder = build_transformer_layer_sequence(heatmap_encoder)
        self.heatmap_decoder = builder.build_head(heatmap_decoder)

    def forward_train(self, img_feats, img_metas, gt_labels_3d, gt_bboxes_3d):
        device = img_feats[0].device

        feat_flatten = []
        spatial_shapes = []
        for scale, feat in enumerate(img_feats):
            feat_flatten.append(feat.flatten(-2))   # (B, N, C, HW)
            spatial_shapes.append(feat.shape[-2:])  # (2,)

        feat_flatten = torch.cat(feat_flatten, -1)     # (B, N, C, ΣHW)
        feat_flatten = feat_flatten.transpose(-2, -1)  # (B, N, ΣHW, C)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.int64, device=device)                      # (S, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros([1,]), spatial_shapes.prod(1).cumsum(0)[:-1]))  # (S,)

        heatmap_feats = self.heatmap_encoder(
            value=feat_flatten,                   # (B, N, ΣHW, C)
            spatial_shapes=spatial_shapes,        # (S, 2)
            level_start_index=level_start_index,  # (S,)
            img_metas=img_metas
        )                                         # (B, Hh*Wh, C)

        heatmap_feats = heatmap_feats.transpose(-2, -1)                                     # (B, C, Hh*Wh)
        heatmap_feats = heatmap_feats.reshape(*heatmap_feats.shape[:-1], self.Hh, self.Wh)  # (B, C, Hh, Wh)

        heatmaps = self.heatmap_decoder(heatmap_feats)
        loss = self.heatmap_decoder.loss(heatmaps, gt_bboxes_3d, gt_labels_3d)

        return loss

    def forward_test(self, img_feats, img_metas, score_threshold):
        device = img_feats[0].device

        feat_flatten = []
        spatial_shapes = []
        for scale, feat in enumerate(img_feats):
            feat_flatten.append(feat.flatten(-2))   # (1, N, C, HW)
            spatial_shapes.append(feat.shape[-2:])  # (2,)

        feat_flatten = torch.cat(feat_flatten, -1)     # (1, N, C, ΣHW)
        feat_flatten = feat_flatten.transpose(-2, -1)  # (1, N, ΣHW, C)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.int64, device=device)                      # (S, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros([1,]), spatial_shapes.prod(1).cumsum(0)[:-1]))  # (S,)

        heatmap_feats = self.heatmap_encoder(
            value=feat_flatten,                   # (1, N, ΣHW, C)
            spatial_shapes=spatial_shapes,        # (S, 2)
            level_start_index=level_start_index,  # (S,)
            img_metas=img_metas
        )                                         # (1, Hh*Wh, C)

        heatmap_feats = heatmap_feats.transpose(-2, -1)                                     # (1, C, Hh*Wh)
        heatmap_feats = heatmap_feats.reshape(*heatmap_feats.shape[:-1], self.Hh, self.Wh)  # (1, C, Hh, Wh)

        heatmaps = self.heatmap_decoder(heatmap_feats)
        heatmaps_by_task = torch.cat([each_task['heatmap'] for each_task in heatmaps], dim=1)  # (1, 10, Hh, Wh)
        heatmaps = heatmaps_by_task.max(1).values.sigmoid()                                    # (1, Hh, Wh)

        heatmaps = F.interpolate(heatmaps[:, None, ...], size=[200, 200], mode='bilinear').squeeze(1)  # (1, 200, 200)
        heatmaps = heatmaps[0].transpose(0, 1)                                                         # (200, 200)

        object_like_coords = (heatmaps > score_threshold).nonzero()  # (n, 2)

        if len(object_like_coords) == 0:
            return None

        return object_like_coords
