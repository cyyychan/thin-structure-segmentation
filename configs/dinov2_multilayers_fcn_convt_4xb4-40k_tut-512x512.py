custom_imports = dict(
    imports=['models', 'datasets', 'hooks'],
    allow_failed_imports=False)

_base_ = [
    './datasets/tut.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py',
]

data_root = '/dataset/siyuanchen/research/data/crack/TUT'
crop_size = (512, 512)

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

# DINOv2 patch=14，需固定 size；size 与 size_divisor 只能二选一
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

# DINOv2 ViT-S/14 多尺度特征 (2,5,8,11) + concat 融合 head
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DinoV2Backbone',
        model_name='dinov2_vits14',  # ViT-S/14, embed_dim=384
        out_indices=(2, 5, 8, 11),
        frozen=True,
    ),
    decode_head=dict(
        type='DinoV2FCNHead',
        in_channels=(384, 384, 384, 384),
        in_index=(0, 1, 2, 3),
        num_classes=2,
        decoder_channels=(256, 128, 64),
        align_corners=False,
        with_edge_attn=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.83),
            dict(
                type='DiceLoss',
                loss_weight=0.17),
        ],
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1,
    ),
    # 三图拼接 Hook：[原图 | GT | 预测]，需配合 --show-dir 使用
    visualization=dict(
        type='SegVisualizationHookConcat3',
        draw=True, 
        interval=1, 
        alpha=0.75, 
        draw_background=False, 
        draw_on_image=False))

# LocalVisBackend 负责在 work_dir 写 .log.json、config、曲线等；TensorBoard 额外写 events
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(vis_backends=vis_backends)

# 额外的 OIS/ODS/mIoU 风格指标（OIS/ODS/mIoU），使用前景 softmax 概率 + 阈值扫描
custom_hooks = [dict(type='MetricsHook', thresh_step=0.01, fg_class=1)]

# DINOv2 backbone 冻结时部分参数不参与 loss，需启用 find_unused_parameters
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)
