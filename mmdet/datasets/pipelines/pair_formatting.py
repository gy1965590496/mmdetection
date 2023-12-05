from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle, Collect, ImageToTensor

@PIPELINES.register_module()
class PairDefaultFormatBundle(DefaultFormatBundle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        # data = {}
        # data.update(outs[0])
        # for k, v in outs[1].items():
        #     data[f'search_{k}'] = v
        return outs

@PIPELINES.register_module()
class PairCollect(Collect):
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

@PIPELINES.register_module()
class PairImageToTensor(ImageToTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs