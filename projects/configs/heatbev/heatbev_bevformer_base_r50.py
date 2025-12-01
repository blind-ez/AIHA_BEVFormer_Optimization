_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False
)

class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 1

sampling_grid_size = [6, 200, 200]

total_epochs = 24
samples_per_gpu = 2
workers_per_gpu = 8
load_from = 'ckpts/bevformer_r50_24ep.pth'

model = dict(
    type='HeatBEV',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe'
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True
    ),
    heatmap_head=dict(
        type='HeatmapHead',
        bev_size=[bev_h_, bev_w_],
        heatmap_encoder=dict(
            type='HeatmapEncoder',
            sampling_grid_size=sampling_grid_size,
            sampling_grid_range=point_cloud_range
        ),
        train_cfg=dict(
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            sampling_grid_size=sampling_grid_size,
            sampling_grid_range=point_cloud_range,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2
        )
    )
)

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='AdjustImageSize', original_img_size=(900, 1600), target_img_size=(256, 704)),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = []

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=samples_per_gpu
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

checkpoint_config = dict(interval=1)
