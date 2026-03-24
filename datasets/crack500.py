# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class Crack500Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'structure'),
        palette=[[0, 0, 0], [6, 230, 230]])

    @classmethod
    def get_label_map(cls, new_classes=None):
        # 0/255 二值图：255→类别 1，必须在 load_data_list 前生效，故用 get_label_map 而非 __init__ 中赋值
        return {0: 0, 255: 1}

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_prefix=data_prefix,
            **kwargs)
