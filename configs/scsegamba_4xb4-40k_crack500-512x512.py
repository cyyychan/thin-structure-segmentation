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

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, size_divisor=None)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SCSegambaBackbone',
        arch='Crack',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        final_norm=True,
        convert_syncbn=True,
        img_size=512,
        in_channels=3,
    ),
    decode_head=dict(
        type='MFSHead',
        in_channels=(128, 64, 32, 16),
        in_index=(0, 1, 2, 3),
        embedding_dim=8,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
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
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1000,
        end=40000,
        by_epoch=False,
    ),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(vis_backends=vis_backends)

