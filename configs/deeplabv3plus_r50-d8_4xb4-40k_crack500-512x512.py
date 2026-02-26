custom_imports = dict(imports=['datasets', 'hooks'], allow_failed_imports=False)

_base_ = [
    './_base_/models/deeplabv3plus_r50-d8.py',
    './datasets/crack500.py', 
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py'
]

data_root = '/dataset/siyuanchen/research/data/crack/Crack500'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

# Fix training input size: pad/resize to 512x512 in data_preprocessor.
# Only set `size` here (no size_divisor).
model = dict(
    data_preprocessor=dict(
        size=(512, 512),
        size_divisor=None))

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