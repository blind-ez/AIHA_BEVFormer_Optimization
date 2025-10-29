# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy

import numpy as np
import torch

from collections import OrderedDict
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.builder import build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from torch.cuda.amp import autocast

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from custom_modules.bbox_utils import denormalize_bbox, predict_bbox_center_from_vel
from custom_modules.coord_utils import lidar_coords_to_bev_coords


@DETECTORS.register_module()
class BEVFormerV2(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 heatmap_head=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 fcos3d_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 num_levels=None,
                 num_mono_levels=None,
                 mono_loss_weight=1.0,
                 frames=(0,),
                 runtime_options=None
                 ):

        super(BEVFormerV2,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        assert not self.fp16_enabled  # not support fp16 yet
        # temporal
        self.video_test_mode = video_test_mode
        assert not self.video_test_mode  # not support video_test_mode yet

        # fcos3d head
        self.fcos3d_bbox_head = build_head(fcos3d_bbox_head) if fcos3d_bbox_head else None
        # loss weight
        self.mono_loss_weight = mono_loss_weight

        # levels of features
        self.num_levels = num_levels
        self.num_mono_levels = num_mono_levels
        self.frames = frames

        self.runtime_options = runtime_options

        self.heatmap_head = build_head(heatmap_head) if heatmap_head is not None else None

        self.prev_frame_info = {
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def get_bev_shift_from_ego_motion(self, img_metas, grid_length=[0.512, 0.512], bev_size=[200, 200]):
        delta_x = np.array([each['can_bus'][0] for each in img_metas])
        delta_y = np.array([each['can_bus'][1] for each in img_metas])
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
        ego_angle = np.array([each['can_bus'][-2] * (180 / np.pi) for each in img_metas])
        bev_angle = ego_angle - translation_angle
        shift_x = translation_length * np.sin(bev_angle * (np.pi / 180)) / grid_length[1] / bev_size[1]
        shift_y = translation_length * np.cos(bev_angle * (np.pi / 180)) / grid_length[0] / bev_size[0]
        shift = torch.tensor([shift_x, shift_y]).transpose(1, 0)
        return shift

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img)
        if 'aug_param' in img_metas[0] and img_metas[0]['aug_param']['CropResizeFlipImage_param'][-1] is True:
            # flip feature 
            img_feats = [torch.flip(x, dims=[-1, ]) for x in img_feats]
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_mono_train(self, img_feats, mono_input_dict):
        """
        img_feats (list[Tensor]): 5-D tensor for each level, (B, N, C, H, W)
        gt_bboxes (list[list[Tensor]]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (list[list[Tensor]]): class indices corresponding to each box
        gt_bboxes_3d (list[list[[Tensor]]): 3D boxes ground truth with shape of
                (num_gts, code_size).
        gt_labels_3d (list[list[Tensor]]): same as gt_labels
        centers2d (list[list[Tensor]]): 2D centers on the image with shape of
                (num_gts, 2).
        depths (list[list[Tensor]]): Depth ground truth with shape of
                (num_gts, ).
        attr_labels (list[list[Tensor]]): Attributes indices of each box.
        img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        ann_idx (list[list[idx]]): indicate which image has mono annotation.
        """
        bsz = img_feats[0].shape[0];
        num_lvls = len(img_feats)

        img_feats_select = [[] for lvl in range(num_lvls)]
        for lvl, img_feat in enumerate(img_feats):
            for i in range(bsz):
                img_feats_select[lvl].append(img_feat[i, mono_input_dict['mono_ann_idx'][i]])
            img_feats_select[lvl] = torch.cat(img_feats_select[lvl], dim=0)
        bsz_new = img_feats_select[0].shape[0]
        assert bsz == len(mono_input_dict['mono_input_dict'])
        input_dict = []
        for i in range(bsz):
            input_dict.extend(mono_input_dict['mono_input_dict'][i])
        assert bsz_new == len(input_dict)
        losses = self.fcos3d_bbox_head.forward_train(img_feats_select, input_dict)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, img_dict, img_metas_dict, **kwargs):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        # Modify: roll back to previous version for single frame
        is_training = self.training
        self.eval()
        prev_bev = OrderedDict({i: None for i in self.frames})
        with torch.no_grad():
            for t in img_dict.keys():
                img = img_dict[t]
                img_metas = [img_metas_dict[t], ]

                if self.runtime_options['run_backbone_fp16']:
                    with autocast():
                        img_feats = self.extract_feat(img=img, img_metas=img_metas)
                        for i, feat in enumerate(img_feats):
                            img_feats[i] = feat.float()
                else:
                    img_feats = self.extract_feat(img=img, img_metas=img_metas)

                if self.num_levels:
                    img_feats = img_feats[:self.num_levels]
                bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev=None, only_bev=True, **kwargs)
                prev_bev[t] = bev.detach()
        if is_training:
            self.train()
        return list(prev_bev.values())

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      gt_bboxes_ignore=None,
                      **mono_input_dict,
                      ):
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]

        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(0)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        img_metas = [img_metas[0], ]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats if self.num_levels is None
                                            else img_feats[:self.num_levels], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)
        losses.update(losses_pts)

        if self.fcos3d_bbox_head:
            losses_mono = self.forward_mono_train(img_feats=img_feats if self.num_mono_levels is None
            else img_feats[:self.num_mono_levels],
                                                  mono_input_dict=mono_input_dict)
            for k, v in losses_mono.items():
                losses[f'{k}_mono'] = v * self.mono_loss_weight

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.sample_idx = 0
        else:
            self.sample_idx += 1
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0][0]['scene_token']

        tmp_pos = copy.deepcopy(img_metas[0][0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0][0]['can_bus'][-1])
        if self.sample_idx != 0:
            img_metas[0][0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0][0]['can_bus'][-1] = 0
            img_metas[0][0][0]['can_bus'][:3] = 0

        kwargs.update(runtime_options=self.runtime_options)
        kwargs.update(frame_cache=dict())

        kwargs['frame_cache'].update(apply_pruning_this_frame=False)

        if self.runtime_options['prune_bev_queries'] and self.runtime_options['prune_based_on_prev_preds']:
            if self.sample_idx != 0 and len(self.prev_frame_info['prev_bbox_preds']) != 0:
                kwargs['frame_cache'].update(prev_bbox_preds=self.prev_frame_info['prev_bbox_preds'])

        outs, bbox_results = self.simple_test(img_metas[0], img[0], prev_bev=None, **kwargs)

        if self.runtime_options['prune_bev_queries'] and self.runtime_options['prune_based_on_prev_preds']:
            cls_scores = outs['all_cls_scores'][-1, 0]
            bbox_preds = outs['all_bbox_preds'][-1, 0]

            preds_mask = cls_scores.max(1).values.sigmoid() > 0.1
            bbox_preds = bbox_preds[preds_mask]

            if kwargs['frame_cache']['apply_pruning_this_frame']:
                bbox_bev_coords = lidar_coords_to_bev_coords(bbox_preds[:, :2], bev_h=200, bev_w=200)
                bbox_bev_idxs = (bbox_bev_coords[:, 1] * 200) + bbox_bev_coords[:, 0]

                preds_mask = (bbox_bev_idxs[:, None] == kwargs['frame_cache']['active_bev_idxs'][None, :]).any(-1)

                bbox_preds = bbox_preds[preds_mask]

            bbox_preds = denormalize_bbox(bbox_preds)
            bbox_preds[:, 2] = bbox_preds[:, 2] - bbox_preds[:, 5] * 0.5

            self.prev_frame_info['prev_bbox_preds'] = bbox_preds

        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False, **kwargs):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, **kwargs)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale, **kwargs)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs, bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_metas)
        img_metas = [img_metas[0], ]

        if self.runtime_options['run_backbone_fp16']:
            with autocast():
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
                for i, feat in enumerate(img_feats):
                    img_feats[i] = feat.float()
        else:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)

        if self.num_levels:
            img_feats = img_feats[:self.num_levels]

        if self.runtime_options['prune_bev_queries'] and self.runtime_options['prune_based_on_heatmap']:
            object_like_coords = self.heatmap_head.forward_test(img_feats, img_metas, self.runtime_options['score_threshold'])
            if object_like_coords is not None:
                kwargs['frame_cache'].update(object_like_coords=object_like_coords)

        if self.runtime_options['prune_bev_queries'] and self.runtime_options['prune_based_on_prev_preds']:
            if 'prev_bbox_preds' in kwargs['frame_cache']:
                shift = self.get_bev_shift_from_ego_motion(img_metas).to(img_feats[0])
                predicted_bbox_centers = predict_bbox_center_from_vel(
                    prev_preds=kwargs['frame_cache']['prev_bbox_preds'],
                    ego_yaw=img_metas[0]['can_bus'][-1] * (np.pi / 180),
                    ego_shift=shift * 200 * 0.512
                )
                kwargs['frame_cache'].update(predicted_bbox_centers=predicted_bbox_centers)

        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas, **kwargs)

        bbox_list = [dict() for i in range(len(img_metas))]
        outs, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale, **kwargs)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return outs, bbox_list
