custom_imports = dict(imports=['datasets', 'hooks'], allow_failed_imports=False)

_base_ = [
    './_base_/models/upernet_swin.py', 
    './datasets/crack500.py',
    './_base_/default_runtime.py', 
    './_base_/schedules/schedule_40k.py'
]

data_root = '/dataset/siyuanchen/research/data/crack/Crack500'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, size_divisor=None)
# backbone 预训练权重：训练时由 init_cfg 自动下载并加载，推理时 init_model 会清掉此处
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    auxiliary_head=dict(in_channels=384, num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

# 三图拼接 Hook：[原图 | GT | 预测]，需配合 --show-dir 使用
# default_hooks = dict(
#     visualization=dict(
#         type='SegVisualizationHookConcat3',
#         draw=True, interval=1,
#         alpha=0.75, draw_background=False))
# LocalVisBackend 负责在 work_dir 写 .log.json、config、曲线等；TensorBoard 额外写 events
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(vis_backends=vis_backends)