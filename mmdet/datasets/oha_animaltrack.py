from .builder import DATASETS
from .coco import CocoDataset
import numpy as np
import torch

@DATASETS.register_module()
class OHAAnimalTrack(CocoDataset):

    CLASSES = ('zebra', 'rabbit', 'pig', 'penguin', 'horse', 
               'goose', 'duck', 'dolphin', 'deer', 'chicken',)

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), 
               (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_occ_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                # gy:add
                gt_occ_labels.append(ann['occ_cls_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            # gy:add
            gt_occ_labels = np.array(gt_occ_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            # gy:add
            gt_occ_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            occ_labels=gt_occ_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
    
    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=None,
    #              metric_items=None):

    #     coco_gt = self.coco
    #     # val_img_ids = [_['id'] for _ in self.data_infos]
    #     # ann_ids = self.coco.get_ann_ids(img_ids=val_img_ids)
    #     # anns = self.coco.load_anns(ann_ids)
    #     # occ_cls_gt = torch.tensor([_['occ_cls_id'] for _ in anns])
    #     # res_len = results.size()[0]
    #     # acc = (torch.sum(results==occ_cls_gt) / res_len).item()
    #     # eval_results = dict()
    #     # eval_results['acc'] = acc
    #     # return eval_results
    #     self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)
    #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
    #     eval_results = self.evaluate_det_occ(results, result_files, coco_gt,
    #                                           metrics, logger, classwise,
    #                                           proposal_nums, iou_thrs,
    #                                           metric_items)
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results
    
