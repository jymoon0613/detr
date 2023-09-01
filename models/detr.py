# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        
        # ! Object queries의 수 
        self.num_queries = num_queries # ! 100

        # ! Transformer (endoer + decoder) 정의
        self.transformer = transformer

        # ! 정의된 Transformer의 feature dimensions
        hidden_dim = transformer.d_model # ! 256

        # ! Class prediction head 정의
        # ! self.class_embed = nn.Linear(256, C + 1)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        # ! Bbox regression head 정의
        # ! self.class_embed = MLP(256, 256, 4, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # ! Object queries 정의
        # ! self.query_embed = nn.Embedding(100, 256) 
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ! Transformer embedding layer 정의
        # ! nn.Conv2d(2048, 256, kernel_size=1)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # ! Backbone 정의
        # ! Backbone은 feature maps 추출을 위한 base CNN (ResNet50)과, 
        # ! 1D flattened된 features에 positional encoding을 더해주는 layer로 구성
        # ! models.backbone 참고
        self.backbone = backbone

        # ! Auxiliary loss 사용 여부
        self.aux_loss = aux_loss # ! True

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # ! samples.tensors = (B, 3, H0, W0) -> 입력 이미지, batch 내의 모든 입력 이미지의 H0, W0는 해당 batch의 maximum H0, maximum W0에 맞게 zero padding됨
        # ! samples.mask    = (B, 3, H0, W0) -> batch 내의 입력 이미지를 동일한 크기로 맞춰주기 위해 zero padding을 추가한 영역에 대한 마스크 (padding된 영역은 True, 나머지는 False)

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # ! Backbone으로부터 feature maps와 positional encodings 추출
        features, pos = self.backbone(samples)
        # ! features = (B, 2048, H, W)의 feature maps와 (B, H, W)의 zero-padding masks의 pairs가 저장된 리스트
        # ! pos      = (B, 256, H, W)의 계산된 positional encodings가 저장된 리스트

        # ! features에 저장된 (feature maps, masks) pairs를 분리
        src, mask = features[-1].decompose()
        # ! src  = (B, 2048, H, W)
        # ! mask = (B, H, W)

        assert mask is not None

        # ! Transformer에 입력
        # ! 우선 feature maps에 대해 Transformer의 input embedding을 계산 -> 채널 수를 2048에서 256으로 축소
        # ! self.input_proj(src)    = (B, 256, H, W) -> embedding된 feature maps
        # ! mask                    = (B, H, W)      -> zero-padding masks
        # ! self.query_embed.weight = (100, 256)     -> 정의된 trainable object queries
        # ! pos[-1]                 = (B, 256, H, W) -> 계산된 positional encodings
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # ! hs[0] = (6, B, 100, 256) -> 각 decoder layer의 output = (B, 100, 256)이 순서대로 총 6개 stack되어 있음
        # ! hs[1] = (B, 256, H, W)   -> 6개의 encoder layers를 통과한 최종 Transformer encoder output
        # ! hs = hs[0]만 선택함 = (6, B, 100, 256)

        # ! Class prediction 수행
        outputs_class = self.class_embed(hs) # ! (6, B, 100, C+1)

        # ! Bbox regression 수행
        outputs_coord = self.bbox_embed(hs).sigmoid() # ! (6, B, 100, 4)

        # ! Output 저장
        # ! 최종 decoder layer의 class prediction 및 bbox regression 결과 저장
        # ! outputs_class[-1] = (B, 100, C+1)
        # ! outputs_coord[-1] = (B, 100, 4)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # ! Auxiliary loss를 사용하는 경우에 대한 처리
        # ! self.aux_loss = True
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            # ! 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output을 저장
            # ! self._set_aux_loss는 {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트를 반환함

        # ! out = 결과 저장 dictionary
        # ! out['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! out['pred_boxes']  = (B, 100, C+1) -> 최종 decoder layer의 bbox prediction
        # ! out['aux_outputs'] = (B, 100, C+1) -> 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
        # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()

        # ! Class의 수 
        self.num_classes = num_classes # ! C

        # ! Bipartite matching을 위한 matcher
        self.matcher = matcher

        # ! Loss 가중치 dictionary
        self.weight_dict = weight_dict

        # ! 'no-object(bg) class'의 classification error에 대한 가중치
        self.eos_coef = eos_coef # ! 0.1

        # ! Loss 항목
        self.losses = losses # ! ['labels', 'boxes', 'cardinality']

        # ! Classification 가중치 적용을 위한 vector 선언
        # ! 'no-object(bg) class'의 경우 0.1
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        # ! Classification loss를 계산함
        # ! outputs = 결과 저장 dictionary
        # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs['pred_boxes']  = (B, 100, 4)   -> 최종 decoder layer의 bbox prediction
        # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
        # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음
        # ! targets = (B,) -> gt_targets
        # ! targets의 경우 전체 B개의 각 이미지마다: 
        # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
        # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
        # ! 가 dictionary 형태로 저장되어 있음
        # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)
        # ! num_boxes = batch 내에 존재하는 총 gt_bboxes의 수를 계산

        assert 'pred_logits' in outputs
        # ! Class 예측값만 추출
        src_logits = outputs['pred_logits'] # ! (B, 100, C+1)

        # ! loss 계산을 위한 index 생성
        idx = self._get_src_permutation_idx(indices)
        # ! idx    = (Bm, Bm)
        # ! idx[0] = (Bm,) -> 0 ~ (B-1)까지의 batch index가 flatten되어 나열되어 있음
        # ! idx[1] = (Bm,) -> 모든 Bm개의 matching indices가 flatten되어 나열되어 있음

        # ! 각 이미지에 존재하는 m개의 gt_label을 matching indices에 따라 정렬한 뒤 나열함
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # ! target_classes_o = (Bm,)

        # ! target값을 생성함
        # ! 먼저 (B, 100) 크기의 tensor를 생성하고, 모든 값을 C로 채움
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # ! 모든 이미지의 100개의 예측값에 gt_classes 할당
        # ! 이때 최적 gt_bboxes와 매칭에 성공한 예측 bboxes는 해당 gt_bboxes의 gt_classes를 할당받는 반면,
        # ! 나머지 예측 bboxes는 그 값이 C로 유지됨
        # ! C는 bg class를 의미하므로, 매칭에 실패한 예측 bboxes는 'no-obj'로 padding된 gt_bboxes와 매칭되는 것이라 볼 수 있음
        target_classes[idx] = target_classes_o # ! (B, 100)

        # ! Cross-entropy를 사용하여 classification loss 계산 (bg class에 대한 weight도 반영)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight) # ! -> scalar loss value

        # ! 계산된 loss를 dictionary에 저장하고 출력
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # ! 추가로 classification error를 계산하고 저장
            # ! src_logits[idx]  = (Bm,) -> gt_bbox를 할당받은 예측값들의 class 예측값만 선택
            # ! target_classes_o = (Bm,) -> 각 이미지에 존재하는 m개의 gt_label을 matching indices에 따라 정렬한 뒤 나열함
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            # ! Top-1 error를 계산하고 저장함

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # ! Bbox regression loss를 계산함
        # ! outputs = 결과 저장 dictionary
        # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs['pred_boxes']  = (B, 100, 4)   -> 최종 decoder layer의 bbox prediction
        # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
        # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음
        # ! targets = (B,) -> gt_targets
        # ! targets의 경우 전체 B개의 각 이미지마다: 
        # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
        # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
        # ! 가 dictionary 형태로 저장되어 있음
        # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)
        # ! num_boxes = batch 내에 존재하는 총 gt_bboxes의 수를 계산

        # ! Class 예측값만 추출
        pred_logits = outputs['pred_logits'] # ! (B, 100, C+1)

        device = pred_logits.device

        # ! 전체 B개의 이미지에 대해 각 이미지마다 몇 개의 gt_bboxes를 포함하는지를 저장
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # ! (B,)
        # Count the number of predictions that are NOT "no-object" (which is the last class)

        # ! 전체 B개의 이미지에 대해 각 이미지마다 'bg'로 예측되지 않은 예측값의 수를 계산함
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1) # !(B,)

        # ! Cardinality error를 계산함
        # ! 'bg'로 예측에 대한 absolute error
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        # ! Bbox regression loss를 계산함
        # ! outputs = 결과 저장 dictionary
        # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs['pred_boxes']  = (B, 100, 4)   -> 최종 decoder layer의 bbox prediction
        # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
        # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음
        # ! targets = (B,) -> gt_targets
        # ! targets의 경우 전체 B개의 각 이미지마다: 
        # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
        # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
        # ! 가 dictionary 형태로 저장되어 있음
        # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)
        # ! num_boxes = batch 내에 존재하는 총 gt_bboxes의 수를 계산

        assert 'pred_boxes' in outputs

        # ! loss 계산을 위한 index 생성
        idx = self._get_src_permutation_idx(indices)
        # ! idx    = (Bm, Bm)
        # ! idx[0] = (Bm,) -> 0 ~ (B-1)까지의 batch index가 flatten되어 나열되어 있음
        # ! idx[1] = (Bm,) -> 모든 Bm개의 matching indices가 flatten되어 나열되어 있음

        # ! (B, 100, 4)의 bbox 예측값 중, gt_bbox와 matching에 성공한 예측값만 선택함
        src_boxes = outputs['pred_boxes'][idx]
        # ! src_boxes = (Bm, 4)

        # ! 각 이미지에 존재하는 m개의 gt_bbox 좌표를 matching indices에 따라 정렬한 뒤 나열함
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # ! target_boxes = (Bm, 4)

        # ! L1 loss 계산하고 저장
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # ! GIoU loss를 계산하고 저장함
        # ! GIoU loss = 1 - GIoU
        # ! box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))는 (Bm, 4)의 예측 bbox와 (Bm, 4)의 gt_bboxes의 bbox 형식을
        # ! 기존 (x, y, w, h)에서 (x1, y1, x2, y2)로 변경하고, GIoU 계산 = (Bn, Bn)
        # ! torch.diag은 2차원 tensor가 주어졌을 때 tensor의 대각 요소를 1D vector로 출력함
        # ! 즉, torch.diag((Bn, Bn))은 매칭된 예측 bbox와 gt_bbox 간의 GIoU를 출력함 = (Bn,)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices

        # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)

        # ! Batch index 생성
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # ! batch_idx = (Bm,) -> 0 ~ (B-1)까지의 batch index가 flatten되어 나열되어 있음
        # ! 이떄 우리는 모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정하므로, 
        # ! 0 ~ m = 0, (m+1) ~ 2m = 1, ...의 형태로 구성되어 있음

        # ! 예측값의 matching index를 추출
        src_idx = torch.cat([src for (src, _) in indices])
        # ! src_idx = (Bm,) -> 모든 Bm개의 matching indices가 flatten되어 나열되어 있음
        # ! 이떄 우리는 모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정하므로, 
        # ! 0 ~ m = 첫 번째 이미지에서 매칭에 성공한 예측값의 indices, (m+1) ~ 2m = 두 번째 이미지에서 매칭에 성공한 예측값의 indices, ...의 형태로 구성되어 있음

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # ! outputs = 결과 저장 dictionary
        # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs['pred_boxes']  = (B, 100, 4)   -> 최종 decoder layer의 bbox prediction
        # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
        # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음
        # ! targets = (B,) -> gt_targets
        # ! targets의 경우 전체 B개의 각 이미지마다: 
        # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
        # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
        # ! 가 dictionary 형태로 저장되어 있음

        # ! 최종 decoder layer의 output만을 추출
        # ! outputs_without_aux['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs_without_aux['pred_boxes']  = (B, 100, C+1) -> 최종 decoder layer의 bbox prediction
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # ! 100개의 predictions와 gt_bboxes의 optimal bipartite matching을 찾음
        indices = self.matcher(outputs_without_aux, targets)
        # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # ! Normalization을 위해 batch 내에 존재하는 총 gt_bboxes의 수를 계산
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        # ! 모든 losses를 계산함
        # ! 결과를 저장할 dictionary 선언
        losses = {}
        for loss in self.losses:

            # ! loss = 계산할 loss의 종류 ('labels', 'boxes', 'cardinality')
            # ! outputs = 결과 저장 dictionary
            # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
            # ! outputs['pred_boxes']  = (B, 100, 4)   -> 최종 decoder layer의 bbox prediction
            # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
            # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음
            # ! targets = (B,) -> gt_targets
            # ! targets의 경우 전체 B개의 각 이미지마다: 
            # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
            # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
            # ! 가 dictionary 형태로 저장되어 있음
            # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)
            # ! num_boxes = batch 내에 존재하는 총 gt_bboxes의 수를 계산
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # ! losses는 'loss_ce', 'class_error', 'loss_bbox', 'loss_giou', 'cardinality_error'를 포함하고 있는 dictionary임
        # ! Key에 대응되는 각 value는 scalar 값임

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # ! Auxiliary loss를 계산하고 저장
        if 'aux_outputs' in outputs: # ! True
            # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
            # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음
            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                # ! aux_outputs = {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)} -> i번째 decoder layer의 output

                # ! 100개의 predictions와 gt_bboxes의 optimal bipartite matching을 찾음
                indices = self.matcher(aux_outputs, targets)
                # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)

                # ! Auxiliary outputs에 대해 모든 losses를 계산함
                for loss in self.losses:

                    if loss == 'masks': # ! False
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                        
                    # ! 'labels' loss 계산 시 log가 True면 Top-1 error를 구하고 logging함
                    # ! 이는 마지막 decoder layer의 출력에만 해당하므로 auxiliary outputs의 경우 logging을 하지 않음 (log=False)
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    # ! loss = 계산할 loss의 종류 ('labels', 'boxes', 'cardinality')
                    # ! aux_outputs = {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)} -> i번째 decoder layer의 output
                    # ! targets = (B,) -> gt_targets
                    # ! targets의 경우 전체 B개의 각 이미지마다: 
                    # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
                    # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
                    # ! 가 dictionary 형태로 저장되어 있음
                    # ! indices = (B, (m,m)) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음 (모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정)
                    # ! num_boxes = batch 내에 존재하는 총 gt_bboxes의 수를 계산
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)

                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # ! losses에 각 auxiliary losses를 추가함
            # ! Auxiliary losses는 'loss_ce_i', 'loss_bbox_i', 'loss_giou_i', 'cardinality_error_i'의 형태로 저장되며, 이떄 i는 decoder layer의 순서임

        # ! losses는 최종 decoder layer(6번째)에 대한 'loss_ce', 'class_error', 'loss_bbox', 'loss_giou', 'cardinality_error'와
        # ! 나머지 5개의 decoder layeres에 대한 'loss_ce_i', 'loss_bbox_i', 'loss_giou_i', 'cardinality_error_i'를 저장하고 있는 dictionary임 (i는 decoder layer의 순서)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    # ! DETR 모델 및 loss 정의
    # ! Classes의 개수 설정
    # ! Num. objects + bg
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    # ! Backbone 정의
    # ! Backbone은 feature maps 추출을 위한 base CNN (ResNet50)과, 
    # ! 1D flattened된 features에 positional encoding을 더해주는 layer로 구성
    # ! models.backbone 참고
    backbone = build_backbone(args)
    # ! backbone = ResNet50 backbone과 position_embedding layer를 연결한 구조

    # ! Transformer (endoer + decoder) 정의
    # ! models.transformer 참고
    transformer = build_transformer(args)
    # ! transformer = 6개의 encoder layers와 6개의 decoder layers로 구성된 Transformer 정의

    # ! 정의된 backbone과 Transformer를 바탕으로 DETR 모델 정의
    # ! num_classes      = C
    # ! args.num_queries = 100
    # ! args.aux_loss    = True
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    if args.masks: # ! False
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # ! Bipartite matching을 위한 matcher 정의
    # ! models.matcher 참고
    matcher = build_matcher(args)

    # ! 각 loss 항목의 가중치 설정
    # ! args.bbox_loss_coef = 5
    # ! args.giou_loss_coef = 2
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.masks: # ! False
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    # ! Auxiliary decoder losses에 대한 가중치 설정
    if args.aux_loss: # ! True
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # ! Loss 항목 정의
    losses = ['labels', 'boxes', 'cardinality']

    if args.masks: # ! False
        losses += ["masks"]

    # ! Loss function 생성
    # ! num_classes   = C (class의 수)
    # ! matcher       = bipartite matching을 위한 matcher
    # ! weight_dict   = loss 가중치 dictionary
    # ! args.eos_coef = 0.1 ('no-object(bg) class'의 classification error에 대한 가중치)
    # ! losses        = loss 항목
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    # ! 예측값을 평가에 적합한 형태로 변경시키기 위해 postprocessors 정의
    postprocessors = {'bbox': PostProcess()}

    if args.masks: # ! False
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
