custom_imports = dict(
    imports=['models', 'datasets', 'hooks'],
    allow_failed_imports=False)

_base_ = [
    './datasets/crack500.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py',
]

data_root = '/dataset/siyuanchen/research/data/crack/Crack500'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

crop_size = (518, 518)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    size_divisor=None,
)

# DINOv2 ViT-S/14 + simple FCN head on Crack500 (512x512)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DinoV2Backbone',
        model_name='dinov2_vits14',  # ViT-S/14, embed_dim=384
        out_indices=(0,),
        frozen=True,
        upsample_to_input=False,     # 让 decode head 负责上采样
    ),
    decode_head=dict(
        type='DinoV2SegHead',
        in_channels=384,     # DINOv2-S embedding dim
        decoder_channels=(256, 128, 64),
        num_classes=2,
        in_index=0,
        align_corners=False,
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

# 三图拼接 Hook：[原图 | GT | 预测]，需配合 --show-dir 使用
# default_hooks = dict(
#     visualization=dict(
#         type='SegVisualizationHookConcat3',
#         draw=True, interval=1,
#         alpha=0.75, draw_background=False, draw_on_image=False))

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

