from mmdet.datasets.builder import PIPELINES
from .transforms import Resize, RandomFlip, Normalize, Pad, BlurAug

@PIPELINES.register_module()
class PairBlurAug(BlurAug):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, results):
        outs = []
        for i, _results in enumerate(results):
            _results = super().__call__(i, _results)
            outs.append(_results)
        return outs    

@PIPELINES.register_module()
class PairResize(Resize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class PairRandomFlip(RandomFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class PairNormalize(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class PairPad(Pad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs