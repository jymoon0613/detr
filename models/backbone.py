# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()

        # ! Backbone 정의
        # ! backbone             = 생성된 pre-trained ResNet50 backbone
        # ! train_backbone       = True
        # ! num_channels         = 2048
        # ! return_interm_layers = False

        for name, parameter in backbone.named_parameters():
            # ! 만약 backbone network를 훈련시키지 않는다면 모든 parameters를 freeze시킴
            # ! Backbone network를 훈련시키는 경우 layer1만 freeze함
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # ! 마지막 layer의 출력만을 사용함
        if return_interm_layers: # ! False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        # ! 생성된 pre-trained ResNet50 backbone의 output 형식을 지정
        # ! IntermediateLayerGetter는 중간 layer의 입력을 추출해주는 역할을 함
        # ! 만약 return_interm_layers = True라면 정의된 return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}에 따라
        # ! self.body의 output은 layer1, layer2, layer3, layer4의 모든 intermediate outputs로 구성되며, dictionary의 형태로 출력됨
        # ! ex. return_layers = {"layer1": "0", ... "layer4": "3"} -> output = {'0': layer1_out_tensors, ..., '3': layer4_out_tensors}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.num_channels = num_channels # ! 2048

    def forward(self, tensor_list: NestedTensor):

        # ! tensor_list.tensors = (B, 3, H0, W0) -> 입력 이미지
        # ! tensor_list.mask    = (B, H0, W0)    -> 입력 이미지에 대한 zero-padding masks

        # ! Backbone으로부터 features 추출
        xs = self.body(tensor_list.tensors)
        # ! xs는 layer4의 output tensor = (B, 2048, H, W)만을 포함하고 있는 dictionary임

        # ! output을 저장할 dictionary 저장
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():

            # ! Backbone 연산을 통해 feature maps는 (H0, W0) -> (H, W)로 subsampling되었지만, masks는 여전히 입력 이미지와 같은 크기이므로,
            # ! Interpolation을 통해 크기를 조절함
            # ! Zero-padding masks의 크기를 (H, W)로 변경함
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # ! (B, H, W)

            # ! Feature maps와 masks를 같이 저장함
            out[name] = NestedTensor(x, mask)

        # ! out = 입력 이미지에 대한 feature maps = (B, 2048, H, W)와 masks = (B, H, W)가 dictionary의 형태로 저장됨
        # ! out은 현재 intermediate features를 사용하지 않으므로 유일한 key = '0'로 구성됨
        
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        
        # ! Pretrained ResNet50 가져오기
        # ! is_main_process() = True
        # ! BatchNorm2D를 batch statistics와 affine parameters가 고정된 FrozenBatchNorm2d로 대체
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        
        # ! Output channel dimension = 2048 (ResNet50)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048 # ! 2048

        # ! Backbone network 정의
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):

        # ! tensor_list.tensors = (B, 3, H0, W0) -> 입력 이미지
        # ! tensor_list.mask    = (B, H0, W0)    -> 입력 이미지에 대한 zero-padding masks

        # ! ResNet50 backbone에서 feature maps 추출
        xs = self[0](tensor_list)
        # ! xs = 입력 이미지에 대한 feature maps = (B, 2048, H, W)와 masks = (B, H, W)가 dictionary의 형태로 저장됨
        # ! xs는 현재 intermediate features를 사용하지 않으므로 유일한 key = '0'로 구성됨

        # ! output 저장을 위한 리스트 생성
        out: List[NestedTensor] = []

        # ! positional encoding 저장을 위한 리스트 생성
        pos = []
        for name, x in xs.items():

            # ! output list에 backbone feature maps = (B, 2048, H, W)와 zero-padding masks = (B, H, W) 저장
            out.append(x)
            # position encoding
            # ! Positional encoding 계산 및 저장
            # ! models.positional_encoding 참고
            pos.append(self[1](x).to(x.tensors.dtype))

        # ! out = (B, 2048, H, W)의 feature maps와 (B, H, W)의 zero-padding masks의 pairs가 저장된 리스트
        # ! pos = (B, 256, H, W)의 계산된 positional encodings가 저장된 리스트

        return out, pos


def build_backbone(args):

    # ! Positional encoding을 더해주는 pos_embed layer 정의
    # ! models.position_encoding 참고
    position_embedding = build_position_encoding(args)

    train_backbone = args.lr_backbone > 0 # ! True
    return_interm_layers = args.masks # ! False

    # ! Backbone CNN 정의
    # ! args.backbone        = 'resnet50'
    # ! train_backbone       = True
    # ! return_interm_layers = False
    # ! args.dilation        = False
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    # ! 정의된 backbone과 position_embedding layer를 연결함
    model = Joiner(backbone, position_embedding)

    model.num_channels = backbone.num_channels # ! 2048

    # ! model = ResNet50 backbone과 position_embedding layer를 연결한 구조

    return model

