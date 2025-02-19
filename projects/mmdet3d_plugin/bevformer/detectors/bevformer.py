import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, len_queue=None):
        B, N, C, H, W = img.size()
        img = img.reshape(-1, C, H, W)
        img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        mlvl_feats = []
        for feat in img_feats:
            _, C, H, W = feat.size()
            if len_queue is not None:
                mlvl_feats.append(feat.view(1, B, N, C, H, W))
            else:
                mlvl_feats.append(feat.view(B, N, C, H, W))

        return mlvl_feats

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        # print('\n\n')
        # print(losses['loss_cls'])
        # print('\n')
        # breakpoint()
        return losses

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas[0], prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(
            pts_feats, img_metas[0], prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self, img_metas, img, **kwargs):
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
            if kwargs['select_queries_based_on_prev_preds']:
                self.prev_frame_info['prev_bbox_preds'] = None
            self.prev_frame_info['sample_idx'] = 0
        else:
            self.prev_frame_info['sample_idx'] += 1

        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])

        if self.prev_frame_info['sample_idx'] == 0:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']

        if kwargs['tracking']:
            if self.prev_frame_info['sample_idx'] == 0:
                self.prev_frame_info['tracking_history'] = dict(prev_bbox_preds=[],
                                                                delta_x=[],
                                                                delta_y=[],
                                                                delta_theta=[],
                                                                length=0)
            else:
                self.prev_frame_info['tracking_history']['delta_x'].append(img_metas[0][0]['can_bus'][0])
                self.prev_frame_info['tracking_history']['delta_y'].append(img_metas[0][0]['can_bus'][1])
                self.prev_frame_info['tracking_history']['delta_theta'].append(img_metas[0][0]['can_bus'][-1])
                if self.prev_frame_info['tracking_history']['length'] < kwargs['tracking_length']:
                    self.prev_frame_info['tracking_history']['length'] += 1
                else:
                    del self.prev_frame_info['tracking_history']['delta_x'][0]
                    del self.prev_frame_info['tracking_history']['delta_y'][0]
                    del self.prev_frame_info['tracking_history']['delta_theta'][0]
                    del self.prev_frame_info['tracking_history']['prev_bbox_preds'][0]
            kwargs['tracking_history'] = self.prev_frame_info['tracking_history']

        if kwargs['select_queries_based_on_prev_preds']:
            if self.prev_frame_info['prev_bbox_preds'] is None or (kwargs['alternate'] and self.prev_frame_info['sample_idx'] % kwargs['alternating_period'] == 0):
                kwargs['apply_occ_mask'] = False
            else:
                kwargs['apply_occ_mask'] = True
                kwargs['occ_reference_coords'] = []
                kwargs['prev_bbox_preds'] = self.prev_frame_info['prev_bbox_preds']
        else:
            kwargs['apply_occ_mask'] = False

        if kwargs['restrict_prev_preds']:
            kwargs['current_occ_mask'] = []

        outs, bbox_result = self.simple_test(img_meta=img_metas[0][0],
                                             img=img[0], # (1, 6, 3, 928, 1600)
                                             prev_bev=self.prev_frame_info['prev_bev'], # (1, 200*200, 256)
                                             **kwargs)

        if kwargs['select_queries_based_on_prev_preds']:
            cls_scores = outs['all_cls_scores'][-1, 0]
            bbox_preds = outs['all_bbox_preds'][-1, 0]
            confidence_scores = cls_scores.max(1).values.sigmoid()
            preds_mask = confidence_scores > kwargs['prev_preds_threshold']
            if preds_mask.sum() == 0:
                self.prev_frame_info['prev_bbox_preds'] = None
            else:
                bbox_preds = bbox_preds[preds_mask]
                if kwargs['restrict_prev_preds']:
                    occ_mask = kwargs['current_occ_mask'][0]
                    if occ_mask is None:
                        self.prev_frame_info['prev_bbox_preds'] = bbox_preds
                    else:
                        bbox_coords = bbox_preds[:, :2]
                        bbox_coords = (bbox_coords - (-51.2)) / 102.4
                        bbox_coords = (bbox_coords * 200).to(torch.int32)
                        bbox_coords_flattened = (bbox_coords[:, 1] * 200) + bbox_coords[:, 0]
                        preds_mask = (bbox_coords_flattened[:, None] == occ_mask[None, :]).any(-1)
                        if preds_mask.sum() == 0:
                            self.prev_frame_info['prev_bbox_preds'] = None
                        else:
                            self.prev_frame_info['prev_bbox_preds'] = bbox_preds[preds_mask]
                else:
                    self.prev_frame_info['prev_bbox_preds'] = bbox_preds

        if kwargs['accurate_mAP']:
            cls_scores = outs['all_cls_scores'][-1, 0]
            bbox_preds = outs['all_bbox_preds'][-1, 0]

            cls_scores, indexs = cls_scores.view(-1).sigmoid().topk(9000)

            labels = indexs % 10

            bbox_index = indexs // 10
            bbox_preds = bbox_preds[bbox_index]

            occ_mask = kwargs['current_occ_mask'][0]
            if occ_mask is not None:
                bbox_coords = bbox_preds[:, :2]
                bbox_coords = (bbox_coords - (-51.2)) / 102.4
                bbox_coords = (bbox_coords * 200).to(torch.int32)
                bbox_coords_flattened = (bbox_coords[:, 1] * 200) + bbox_coords[:, 0]

                preds_mask = (bbox_coords_flattened[:, None] == occ_mask[None, :]).any(-1)
                if preds_mask.sum() != 0:
                    cls_scores = cls_scores[preds_mask]
                    normalized_bboxes = bbox_preds[preds_mask]
                    labels = labels[preds_mask]

                    rot_sine = normalized_bboxes[..., 6:7]
                    rot_cosine = normalized_bboxes[..., 7:8]
                    rot = torch.atan2(rot_sine, rot_cosine)
                    cx = normalized_bboxes[..., 0:1]
                    cy = normalized_bboxes[..., 1:2]
                    cz = normalized_bboxes[..., 4:5]
                    w = normalized_bboxes[..., 2:3]
                    l = normalized_bboxes[..., 3:4]
                    h = normalized_bboxes[..., 5:6]
                    w = w.exp() 
                    l = l.exp() 
                    h = h.exp() 
                    vx = normalized_bboxes[:, 8:9]
                    vy = normalized_bboxes[:, 9:10]
                    denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)

                    if len(labels) > 300:
                        cls_scores = cls_scores[:300]
                        denormalized_bboxes = denormalized_bboxes[:300]
                        labels = labels[:300]

                    bboxes = img_metas[0][0]['box_type_3d'](denormalized_bboxes.cpu(), 9)

                    bbox_result[0]['pts_bbox']['scores_3d'] = cls_scores.cpu()
                    bbox_result[0]['pts_bbox']['labels_3d'] = labels.cpu()
                    bbox_result[0]['pts_bbox']['boxes_3d'] = bboxes

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = outs['bev_query']
        if kwargs['tracking']:
            self.prev_frame_info['tracking_history']['prev_bbox_preds'].append(self.prev_frame_info['prev_bbox_preds'])

        if kwargs['log']:
            return bbox_result, kwargs['logger']
        else:
            return bbox_result

    def simple_test(self, img_meta, img, prev_bev, rescale=False, **kwargs):
        mlvl_feats = self.extract_feat(img=img) # (1, 6, 256, H/8, W/8), (1, 6, 256, H/16, W/16), (1, 6, 256, H/32, W/32), (1, 6, 256, H/64, W/64)

        outs = self.pts_bbox_head(mlvl_feats=mlvl_feats, prev_bev=prev_bev, img_meta=img_meta, **kwargs)

        bboxes, scores, labels = self.pts_bbox_head.get_bboxes(preds_dicts=outs, img_meta=img_meta, rescale=rescale)
        bbox_result = [{'pts_bbox': bbox3d2result(bboxes, scores, labels)}]

        return outs, bbox_result