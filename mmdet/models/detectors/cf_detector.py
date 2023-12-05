from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..builder import build_roi_extractor
from mmdet.core import bbox2roi, bbox2result

@DETECTORS.register_module()
class CFDetector(TwoStageDetector):

    def __init__(self,
                 backbone,
                 bbox_roi_extractor,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(CFDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.query_roi_extractor = build_roi_extractor(bbox_roi_extractor)
    

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, 
                      search_img, search_img_metas, search_gt_bboxes, search_gt_labels,
                      gt_bboxes_ignore=None, search_gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        z_feat = self.extract_feat(img)
        x_feat = self.extract_feat(search_img)
        rois = bbox2roi(gt_bboxes)
        query_feats = self.query_roi_extractor(z_feat, rois)#torch.Size([13, 256, 7, 7])

        losses = dict()
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_feat,
                search_img_metas,
                search_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=search_gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(x_feat, query_feats, search_img_metas, proposal_list,
                                                 search_gt_bboxes, search_gt_labels,
                                                 search_gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

        

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_metas, gt_bboxes, 
                    search_img, search_img_metas, search_gt_bboxes,
                    proposals=None, rescale=False):
        """Test without augmentation."""
        if isinstance(search_img, list):
            search_img = search_img[0]
        gt_bboxes = [gt_bbox.squeeze(0) for gt_bbox in gt_bboxes]

        assert self.with_bbox, 'Bbox head must be implemented.'
        z_feat = self.extract_feat(img)
        x_feat = self.extract_feat(search_img)
        rois = bbox2roi(gt_bboxes)
        query_feats = self.query_roi_extractor(z_feat, rois)#torch.Size([1, 256, 7, 7])
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x_feat, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x_feat, img_metas, query_feats, proposal_list, rescale=rescale)
    
