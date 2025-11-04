import torch

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from torch.cuda.amp import autocast

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class HeatBEV(MVXTwoStageDetector):
    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 heatmap_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.heatmap_head = builder.build_head(heatmap_head)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None):
        len_queue = img.shape[1]
        img = img[:, -1, :, :, :, :]  # (B, N, 3, H, W)
        img_metas = [each[len_queue-1] for each in img_metas]

        self.eval()
        with torch.no_grad():
            img_feats = self.extract_feat(img=img)
        self.train()
        for i, feat in enumerate(img_feats):
            img_feats[i] = feat.float()

        losses = dict()

        loss_heatmap = self.heatmap_head.forward_train(img_feats, img_metas, gt_labels_3d, gt_bboxes_3d)
        losses.update(loss_heatmap)

        return losses

    @autocast()
    def extract_feat(self, img, len_queue=None):
        B = img.size(0)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

        img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped
