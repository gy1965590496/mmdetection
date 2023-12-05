from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class AnimalTrack(CocoDataset):

    CLASSES = ('zebra', 'rabbit', 'pig', 'penguin', 'horse', 
               'goose', 'duck', 'dolphin', 'deer', 'chicken',)

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), 
               (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

