# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose
from .test_time_aug import MultiScaleFlipAug

@PIPELINES.register_module()
class PairMultiScaleFlipAug(MultiScaleFlipAug):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        data = {}
        data.update(outs[0])
        for k, v in outs[1].items():
            data[f'search_{k}'] = v
        return data

