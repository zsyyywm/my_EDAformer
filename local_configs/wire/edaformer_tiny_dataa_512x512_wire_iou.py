# EDAFormer-Tiny + 电线二分类 DataA（与 TransNeXt ``mask2former_transnext_tiny_dataa_512x512_iou.py`` 对齐）
#
# 推荐训练（方案一；与 MambaVision 单入口对齐亦可）::
#   python tools/train.py --wire-scheme dataa
#   python tools/train_edaformer_wire_dataa.py
# 测试::
#   python tools/test_edaformer_wire_dataa.py
#
# 等价项：200 epoch / val 每 epoch / 前景 IoU 存 best / 早停 patience=50 / gloo /
# ``data/checkpoints1/train_<时间>/`` 下放日志与权重；``val_metrics.csv`` + 曲线图。
# 环境变量：``WIRE_SEG_DATA_ROOT``（DataA-B 父目录）、``WIRE_SEG_DATAA_ROOT``（直指 DataA）。
#
# 数据：reapear.md — ``DataA-B/DataA/image/{train,val,test}`` 与 ``mask/...``

wire_seg_experiment = True

_base_ = [
    '../_base_/models/edaformer.py',
    '../_base_/default_runtime.py',
]

# 单卡非 DDP 时 SyncBN 会报「process group has not been initialized」；电线脚本默认单卡用 BN
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
# 增广：TransNeXt 线为 RandomChoiceResize+短边多尺度；此处用 mmseg0.13 的 ratio_range 随机缩放近似
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

_data_root = '../../DataA-B/DataA'
_class_names = ('background', 'foreground')
_palette = [[0, 0, 0], [255, 255, 255]]

model = dict(
    # 官方网盘常见文件名为 ``EFT_b0.pth``；若已重命名为 ``EFT_T.pth`` 可改回
    pretrained='EFT_b0.pth',
    backbone=dict(
        type='EFT_T',
        style='pytorch',
        reduction_ratios=[1, 1, 1, 1]),
    decode_head=dict(
        type='EDAFormerHead',
        in_channels=[64, 128, 256],
        in_index=[1, 2, 3],
        mlp_ratio=2,
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        reduction_ratios=[1, 1, 1],
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=128),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer_config = dict()
# 与 DataA 配置中 PolyLR（无 warmup）一致
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0,
    by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(
    by_epoch=True,
    interval=10**9,
    save_last=False)
log_config = dict(
    interval=10,
    hooks=[dict(type='WireCompactTextLoggerHook', by_epoch=True)])

evaluation = dict(
    interval=1,
    metric='mIoU',
    save_best='IoU',
    rule='greater')

# 与 TransNeXt 配置键名一致，便于对照 reapear.md
mask2former_iou_early_stop_patience = 50
custom_hooks = [
    dict(type='WireEpochBannerHook', priority=30),
    dict(type='WireTrainEpochEndBlockHook', priority=92),
    dict(type='WireConsoleSummaryHook', priority=55),
    dict(
        type='WireIoUPatienceEarlyStopHook',
        priority=65,
        monitor='IoU',
        patience=mask2former_iou_early_stop_patience,
        rule='greater'),
    dict(type='WirePlotMetricsHook', priority=75, sample_interval=50),
]

dist_params = dict(backend='gloo')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='WireBinaryDataset',
        data_root=_data_root,
        img_dir='image/train',
        ann_dir='mask/train',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=list(_class_names),
        palette=list(_palette),
        pipeline=train_pipeline),
    val=dict(
        type='WireBinaryDataset',
        data_root=_data_root,
        img_dir='image/val',
        ann_dir='mask/val',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=list(_class_names),
        palette=list(_palette),
        pipeline=test_pipeline),
    test=dict(
        type='WireBinaryDataset',
        data_root=_data_root,
        img_dir='image/test',
        ann_dir='mask/test',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=list(_class_names),
        palette=list(_palette),
        pipeline=test_pipeline))

work_dir = 'data/checkpoints1'
