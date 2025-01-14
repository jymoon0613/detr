# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        # ! samples.tensors = (B, 3, H0, W0) -> 입력 이미지, batch 내의 모든 입력 이미지의 H0, W0는 해당 batch의 maximum H0, maximum W0에 맞게 zero padding됨
        # ! samples.mask    = (B, 3, H0, W0) -> batch 내의 입력 이미지를 동일한 크기로 맞춰주기 위해 zero padding을 추가한 영역에 대한 마스크 (padding된 영역은 True, 나머지는 False)
        # ! targets         = (B,)           -> targets
        # ! targets의 경우 전체 B개의 각 이미지마다: 
        # ! (1) gt_bboxes의 좌표 = (n,4), (2) gt_bboxes의 class labels = (n,), (3) image_id = (1,), 
        # ! (4) 각 gt_bbox의 크기 = (n,), (5) 이미지에 많은 개체가 포함되어 있는지 여부 = (n,), (6) 원본 이미지 크기 = (HH,WW), (7) 현재 이미지 크기 = (H0,W0)
        # ! 가 dictionary 형태로 저장되어 있음 (각 이미지마다 n개의 gt_bboxes가 존재한다고 가정)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ! model에 입력하여 예측값 계산
        # ! models.detr 참고
        # ! samples.tensors = (B, 3, H0, W0) -> 입력 이미지, batch 내의 모든 입력 이미지의 H0, W0는 해당 batch의 maximum H0, maximum W0에 맞게 zero padding됨
        # ! samples.mask    = (B, 3, H0, W0) -> batch 내의 입력 이미지를 동일한 크기로 맞춰주기 위해 zero padding을 추가한 영역에 대한 마스크 (padding된 영역은 True, 나머지는 False)
        outputs = model(samples)
        # ! outputs = 결과 저장 dictionary
        # ! outputs['pred_logits'] = (B, 100, C+1) -> 최종 decoder layer의 class prediction
        # ! outputs['pred_boxes']  = (B, 100, 4)   -> 최종 decoder layer의 bbox prediction
        # ! outputs['aux_outputs'] = 최종 decoder layer를 제외한 나머지 5개의 decoder layer의 output이 
        # ! -> {'pred_logits': (B, 100, C+1), 'pred_boxes': (B, 100, 4)}의 dictionary가 총 5개 포함된 리스트로 저장되어 있음

        # ! Loss 계산
        # ! outputs = 결과 저장 dictionary (위 참고)
        # ! targets = (B,) -> gt_targets
        loss_dict = criterion(outputs, targets)
        # ! loss_dict는 최종 decoder layer(6번째)에 대한 'loss_ce', 'class_error', 'loss_bbox', 'loss_giou', 'cardinality_error'와
        # ! 나머지 5개의 decoder layeres에 대한 'loss_ce_i', 'loss_bbox_i', 'loss_giou_i', 'cardinality_error_i'를 저장하고 있는 dictionary임 (i는 decoder layer의 순서)

        # ! Loss 가중치 dictionary 추출
        weight_dict = criterion.weight_dict

        # ! 가중치 반영하여 total loss계산
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
