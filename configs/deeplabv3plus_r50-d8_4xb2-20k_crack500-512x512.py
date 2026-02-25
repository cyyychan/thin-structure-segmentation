custom_imports = dict(imports=['datasets'], allow_failed_imports=False)

_base_ = [
    './_base_/models/deeplabv3plus_r50-d8.py',
    './datasets/crack500.py', 
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]

data_root = '/dataset/siyuanchen/research/data/crack/Crack500'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))

# Fix training input size: pad/resize to 512x512 in data_preprocessor.
# Only set `size` here (no size_divisor).
model = dict(
    data_preprocessor=dict(
        size=(512, 512),
        size_divisor=None))

# log scalars/images to TensorBoard
vis_backends = [
    dict(type='TensorboardVisBackend')
]
visualizer = dict(vis_backends=vis_backends)