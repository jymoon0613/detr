# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # ! 1
        self.cost_bbox = cost_bbox # ! 5
        self.cost_giou = cost_giou # ! 2
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs['pred_boxes']  = (B, 100, C+1) -> 최종 decoder layer의 bbox prediction
        # ! targets = (B,) -> gt_targets
        # ! targets의 경우 전체 B개의 각 이미지마다: 
        # ! (1) gt_bboxes의 좌표, (2) gt_bboxes의 class labels, (3) image_id, 
        # ! (4) 각 gt_bbox의 크기, (5) 이미지에 많은 개체가 포함되어 있는지 여부, (6) 원본 이미지 크기, (7) 현재 이미지 크기
        # ! 가 dictionary 형태로 저장되어 있음

        bs, num_queries = outputs["pred_logits"].shape[:2] # ! B, 100

        # We flatten to compute the cost matrices in a batch
        # ! 예측값 reshape
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) # ! (100B, C+1) [batch_size * num_queries, num_classes] 
        out_bbox = outputs["pred_boxes"].flatten(0, 1) # ! (100B, 4) [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # ! Target 값도 reshape함
        # ! 하나의 배치에 총 n개의 gt_bboxes가 존재한다고 가정함
        tgt_ids = torch.cat([v["labels"] for v in targets]) # ! (n,) 
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # ! (n,4) 

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # ! Class에 대한 cost 계산
        # ! 모든 100B개의 예측값에 대해 모든 n개의 gt_bboxes의 gt_classes에 대한 classification score를 추출함
        # ! '-' 를 붙여서 가장 높은 classification score를 보이는 gt_class가 가장 낮은 cost를 지니도록 함
        cost_class = -out_prob[:, tgt_ids]
        # ! cost_class = (100B, n)

        # Compute the L1 cost between boxes
        # ! Bbox에 대한 L1 cost 계산
        # ! 모든 100B개의 예측값에 대해 모든 n개의 gt_bboxes와의 L1 distance를 계산함
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # ! cost_bbox = (100B, n)

        # Compute the giou cost betwen boxes
        # ! Bbox에 대한 GIoU cost 계산
        # ! 모든 100B개의 예측값에 대해 모든 n개의 gt_bboxes와의 GIoU를 계산함
        # ! 먼저 예측 bbox와 gt_bbox의 bbox 형식을 (x, y, w, h)에서 (x1, y1, x2, y2)로 변경하고 GIoU를 계산함
        # ! Classification cost와 마찬가지로 '-' 를 붙여서 가장 높은 GIoU score를 보이는 gt_bbox가 가장 낮은 cost를 지니도록 함
        # ! util.box_ops 참고
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # ! cost_giou = (100B, n)

        # Final cost matrix
        # ! 각 cost 항을 weighted sum하여 최종 cost matrix 정의
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # ! (100B, n)
        C = C.view(bs, num_queries, -1).cpu() # ! (B, 100, n)

        # ! 전체 B개의 각 이미지마다 몇 개의 gt_bboxes가 존재하는지 저장
        sizes = [len(v["boxes"]) for v in targets] # ! (B,)

        # ! 전체 B개의 각 이미지마다 100개의 예측값과 해당 이미지의 m개의 gt_bboxes 간의 최적 matching을 찾음
        # ! C.split(sizes, -1)은 (B, 100, n)의 cost matrix에서 n개의 총 gt_bboxes를 각 이미지에 존재하는 m개의 gt_bboxes 단위로 분할함
        # ! 만약 i번째 이미지에 m개의 gt_bboxes가 존재한다면, i번째 반복에서 c = (B, 100, m)
        # ! 또한 c = (B, 100, m)에서 i번째 행을 선택하여 i번째 이미지에 대한 예측값만이 고려되도록 함 (c[i] = (100, m))
        # ! 선택된 i번째 iteration의 cost sub_matrix에 대해 linear_sum_assignment으로 최적의 matching을 결정함
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # ! indices = (B,) -> 전체 B개의 각 이미지마다 최적 matching의 결과가 리스트에 저장되어 있음
        # ! 최적 matching 결과는 예측 bbox indices와 각각에 대응되는 최적 gt_bbox indices가 tuple 형태로 저장되어 있음
        # ! 이때 matching 결과로 출력되는 예측 bbox indices의 길이와 최적 gt_bbox indices의 길이는 같으며, 100개의 예측 bboxes와 m개의 gt_bboxes 중 더 작은 크기로 고정됨
        # ! 보통 예측 bboxes보다 gt_bboxes의 수가 훨씬 작으므로 각 이미지마다 gt_bboxes의 수만큼 출력될 것임 (gt_bboxes의 수는 각 이미지마다 상이하다는 점에 주의)
        # ! 따라서, 모든 이미지에 정확히 m개의 gt_bboxes가 존재한다고 가정한다면 각 이미지의 matching 결과는 (m,m)의 tuple일 것임

        # ! 결과를 tensor로 변경하여 출력
         
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):

    # ! Bipartite matching을 위한 matcher 정의
    # ! args.set_cost_class = 1
    # ! args.set_cost_bbox  = 5
    # ! args.set_cost_giou  = 2
    # ! -> 각각 class, bbox, giou를 matching에 얼마나 반영할 것인지에 관한 coefficient
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
