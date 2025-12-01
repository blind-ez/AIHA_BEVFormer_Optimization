import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.builder import HEADS, build_head, build_loss
from mmdet3d.models.utils import clip_sigmoid


@HEADS.register_module()
class HeatmapHead(BaseModule):
    def __init__(self,
                 bev_size=None,
                 num_scales=None,
                 heatmap_encoder=None,
                 heatmap_decoder=None,
                 train_cfg=None):
        super().__init__()

        self.Hb, self.Wb = bev_size

        self.scale_fusion = nn.Sequential(
            nn.Conv2d(256*num_scales, 256, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
        )

        self.z_fusion = nn.Sequential(
            nn.Conv2d(256*6, 256, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(256),
            nn.ReLU()
        )

        self.heatmap_encoder = build_head(heatmap_encoder)

        self.heatmap_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        )

        self.train_cfg = train_cfg
        self.loss_cls = build_loss(train_cfg['loss_cls'])

    def unify_feature_scales(self, multi_scale_feats, target_scale):
        B, N, C, _, _ = multi_scale_feats[0].shape

        aligned_feats = list()
        for scale, feat in enumerate(multi_scale_feats):
            feat = feat.flatten(0, 1)
            aligned_feat = F.interpolate(feat, size=target_scale, mode='bilinear')
            aligned_feat = aligned_feat.reshape(B, N, *aligned_feat.shape[1:])
            aligned_feats.append(aligned_feat)

        return aligned_feats

    def forward_test(self, img_feats, img_metas, score_threshold):
        B, N, C, H, W = img_feats[0].shape

        assert B == 1

        aligned_feats = self.unify_feature_scales(img_feats, (H, W))
        aligned_feats = torch.cat(aligned_feats, dim=2)                                       # (1, N, S*C, H, W)
        fused_feats = self.scale_fusion(aligned_feats.flatten(0, 1)).reshape(B, N, -1, H, W)  # (1, N, Ch, H, W)

        volume_embed = self.heatmap_encoder(fused_feats, img_metas)  # (1, Ch, Dh, Hh, Wh)
        bev_embed = self.z_fusion(volume_embed.flatten(1, 2))        # (1, Ch, Hh, Wh)

        heatmap = self.heatmap_decoder(bev_embed)  # (1, 1, Hh, Wh)

        heatmap = heatmap.sigmoid()
        heatmap = F.interpolate(heatmap, size=[self.Hb, self.Wb], mode='bilinear')  # (1, 1, Hb, Wb)
        heatmap = heatmap[0, 0].transpose(0, 1)                                     # (Wb, Hb)

        object_like_coords = (heatmap > score_threshold).nonzero()  # (n, 2)

        if len(object_like_coords) == 0:
            return None

        return object_like_coords

    def forward_train(self, img_feats, img_metas, gt_labels_3d, gt_bboxes_3d):
        B, N, C, H, W = img_feats[0].shape

        aligned_feats = self.unify_feature_scales(img_feats, (H, W))
        aligned_feats = torch.cat(aligned_feats, dim=2)                                       # (B, N, S*C, H, W)
        fused_feats = self.scale_fusion(aligned_feats.flatten(0, 1)).reshape(B, N, -1, H, W)  # (B, N, Ch, H, W)

        volume_embed = self.heatmap_encoder(fused_feats, img_metas)  # (B, Ch, Dh, Hh, Wh)
        bev_embed = self.z_fusion(volume_embed.flatten(1, 2))        # (B, Ch, Hh, Wh)

        heatmap = self.heatmap_decoder(bev_embed)  # (B, 1, Hh, Wh)

        loss = self.loss(heatmap, gt_bboxes_3d, gt_labels_3d)

        return loss

    def loss(self, pred_heatmaps, gt_bboxes_3d, gt_labels_3d):
        pred = clip_sigmoid(pred_heatmaps)
        target = self.get_targets(gt_bboxes_3d, gt_labels_3d)

        num_pos = (target==1).sum().item()

        loss_heatmap = self.loss_cls(pred.flatten(0, 1), target.flatten(0, 1), avg_factor=max(num_pos, 1))

        loss_dict = dict()        
        loss_dict[f'loss_heatmap'] = loss_heatmap

        return loss_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        B = len(gt_bboxes_3d)

        target_heatmaps = list()
        for batch in range(B):
            target_heatmap = self.get_targets_single(gt_bboxes_3d[batch], gt_labels_3d[batch])
            target_heatmaps.append(target_heatmap.unsqueeze(0))

        target_heatmaps = torch.stack(target_heatmaps, dim=0)

        return target_heatmaps

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(gt_labels_3d.device)

        grid_size = self.train_cfg['sampling_grid_size']
        grid_range = self.train_cfg['sampling_grid_range']
        voxel_size = [(grid_range[3] - grid_range[0]) / grid_size[2], (grid_range[4] - grid_range[1]) / grid_size[1]]

        target_heatmap = gt_bboxes_3d.new_zeros([grid_size[1], grid_size[2]])

        num_objs = min(len(gt_bboxes_3d), self.train_cfg['max_objs'])

        for k in range(num_objs):
            width = gt_bboxes_3d[k][3] / voxel_size[0]
            length = gt_bboxes_3d[k][4] / voxel_size[1]

            if not ((width > 0) and (length > 0)):
                continue

            radius = gaussian_radius([length, width], min_overlap=self.train_cfg['gaussian_overlap'])
            radius = max(self.train_cfg['min_radius'], int(radius))

            x = gt_bboxes_3d[k][0]
            y = gt_bboxes_3d[k][1]

            coord_x = (x - grid_range[0]) / voxel_size[0]
            coord_y = (y - grid_range[1]) / voxel_size[1]

            center = gt_bboxes_3d.new_tensor([coord_x, coord_y]).long()

            if not ((0 <= center[0] < grid_size[2]) and (0 <= center[1] < grid_size[1])):
                continue

            draw_heatmap_gaussian(target_heatmap, center, radius)

        return target_heatmap
