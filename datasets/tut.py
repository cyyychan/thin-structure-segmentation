# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class TUTDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'structure'),
        palette=[[0, 0, 0], [255, 255, 255]])

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
        # 0/255 二值图：255→类别 1，避免 reduce_zero_label 把 255 当 ignore
        self.label_map = {0: 0, 255: 1}
