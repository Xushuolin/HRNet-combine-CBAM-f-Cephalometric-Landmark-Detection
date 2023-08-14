default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        max_keep_ckpts=5,
        save_best='SDR 2.0mm',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,
        show=False,
        out_dir='E:\\CL-Detection2023-MMPose\\MMPose-visualize',
        interval=1))
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = 'E:\\CL-Detection2023-MMPose\\MMPose-checkpoints\\best_SDR 2.0mm_epoch_170.pth'
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=2)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[
            170,
            200,
        ],
        gamma=0.1,
        by_epoch=True),
]
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='MSRAHeatmap',
    input_size=(
        512,
        512,
    ),
    heatmap_size=(
        128,
        128,
    ),
    sigma=2)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_channels=(
                    48,
                    96,
                )),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(
                    5,
                    5,
                    5,
                ),
                num_channels=(
                    48,
                    96,
                    192,
                )),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_channels=(
                    48,
                    96,
                    192,
                    384,
                ))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth'
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=38,
        deconv_out_channels=None,
        loss=dict(type='KeypointOHKMMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(
                512,
                512,
            ),
            heatmap_size=(
                128,
                128,
            ),
            sigma=2)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
dataset_type = 'CephalometricDataset'
data_mode = 'topdown'
data_root = 'E:/CL-Detection2023-MMPose/CL-detection2023'
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform', shift_prob=0, rotate_factor=60),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.2,
                max_width=0.2,
                min_holes=1,
                min_height=0.1,
                min_width=0.1,
                p=1.0),
        ]),
    dict(type='TopdownAffine', input_size=(
        512,
        512,
    )),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='MSRAHeatmap',
            input_size=(
                512,
                512,
            ),
            heatmap_size=(
                128,
                128,
            ),
            sigma=2)),
    dict(
        type='PackPoseInputs',
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'category_id',
            'crowd_index',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
            'flip',
            'flip_direction',
            'flip_indices',
            'raw_ann_info',
            'spacing',
        )),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(
        512,
        512,
    )),
    dict(
        type='PackPoseInputs',
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'category_id',
            'crowd_index',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
            'flip',
            'flip_direction',
            'flip_indices',
            'raw_ann_info',
            'spacing',
        )),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CephalometricDataset',
        data_root='E:/CL-Detection2023-MMPose/CL-detection2023',
        data_mode='topdown',
        ann_file='train.json',
        data_prefix=dict(img='MMPose/'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale', padding=1.0),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomBBoxTransform', shift_prob=0, rotate_factor=60),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.2,
                        max_width=0.2,
                        min_holes=1,
                        min_height=0.1,
                        min_width=0.1,
                        p=1.0),
                ]),
            dict(type='TopdownAffine', input_size=(
                512,
                512,
            )),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(
                        512,
                        512,
                    ),
                    heatmap_size=(
                        128,
                        128,
                    ),
                    sigma=2)),
            dict(
                type='PackPoseInputs',
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'category_id',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'spacing',
                )),
        ]))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CephalometricDataset',
        data_root='E:/CL-Detection2023-MMPose/CL-detection2023',
        data_mode='topdown',
        ann_file='valid.json',
        data_prefix=dict(img='MMPose/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(
                512,
                512,
            )),
            dict(
                type='PackPoseInputs',
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'category_id',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'spacing',
                )),
        ]))
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CephalometricDataset',
        data_root='E:/CL-Detection2023-MMPose/CL-detection2023',
        data_mode='topdown',
        ann_file='test.json',
        data_prefix=dict(img='MMPose/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(
                512,
                512,
            )),
            dict(
                type='PackPoseInputs',
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'category_id',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'spacing',
                )),
        ]))
val_evaluator = dict(type='CephalometricMetric')
test_evaluator = dict(type='CephalometricMetric')
launcher = 'none'
work_dir = './work_dirs\\td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss'
