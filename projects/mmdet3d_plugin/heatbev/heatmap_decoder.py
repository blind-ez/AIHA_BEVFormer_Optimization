import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid


@HEADS.register_module()
class HeatmapDecoder(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 tasks,
                 separate_head,
                 loss_cls,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]

        self.shared_conv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            bias='auto'
        )

        self.task_heads = nn.ModuleList()
        for num_cls in num_classes:
            heads = dict()
            heads.update(dict(heatmap=(num_cls, 2)))
            separate_head.update(in_channels=mid_channels, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

        self.loss_cls = build_loss(loss_cls)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, x):
        x = self.shared_conv(x)

        ret_dicts = []
        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def loss(self, preds_dicts, gt_bboxes_3d, gt_labels_3d, **kwargs):
        heatmaps = self.get_targets(gt_bboxes_3d, gt_labels_3d)

        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            pred = clip_sigmoid(preds_dict['heatmap'])
            target = heatmaps[task_id]
            num_pos = (target==1).sum().item()
            loss_heatmap = self.loss_cls(pred.flatten(0, 1), target.flatten(0, 1), avg_factor=max(num_pos, 1))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap

        return loss_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        heatmaps = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        heatmaps = [torch.stack(each_task) for each_task in heatmaps]

        return heatmaps

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(gt_labels_3d.device)

        grid_size = self.train_cfg['bev_grid_size']
        pc_range = self.train_cfg['pc_range']
        voxel_size = [(pc_range[3] - pc_range[0]) / grid_size[1], (pc_range[4] - pc_range[1]) / grid_size[0]]

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([torch.where(gt_labels_3d == class_name.index(i) + flag) for i in class_name])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append((gt_labels_3d[m] + 1) - flag2)
            task_boxes.append(torch.cat(task_box, dim=0))
            task_classes.append(torch.cat(task_class))
            flag2 += len(mask)

        heatmaps = []
        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros((len(self.class_names[idx]), grid_size[0], grid_size[1]))

            num_objs = min(task_boxes[idx].shape[0], self.train_cfg['max_objs'])

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3] / voxel_size[0]
                length = task_boxes[idx][k][4] / voxel_size[1]

                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    x, y = task_boxes[idx][k][0], task_boxes[idx][k][1]

                    coord_x = (x - pc_range[0]) / voxel_size[0]
                    coord_y = (y - pc_range[1]) / voxel_size[1]

                    center = gt_bboxes_3d.new_tensor([coord_x, coord_y])
                    center_int = center.long()

                    if not (0 <= center_int[0] < grid_size[1] and 0 <= center_int[1] < grid_size[0]):
                        continue

                    draw_heatmap_gaussian(heatmap[cls_id], center_int, radius)

            heatmaps.append(heatmap)

        return heatmaps
