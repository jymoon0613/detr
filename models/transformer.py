# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # ! d_model            = 256
        # ! dropout            = 0.1
        # ! nhead              = 8
        # ! dim_feedforward    = 2048
        # ! num_encoder_layers = 6
        # ! num_decoder_layers = 6
        # ! normalize_before   = False

        # ! Transformer encoder 정의
        # ! 먼저 encoder layer 구조를 정의
        # ! encoder_layer = TransformerEncoderLayer(256, 8, 2048, 0.1, "relu", False)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None # ! None

        # ! 정의된 encoder layer 구조를 바탕으로 encoder 정의
        # ! self.encoder = TransformerEncoder(encoder_layer, 6, None)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # ! Transformer decoder 정의
        # ! 먼저 decoder layer 구조를 정의
        # ! decoder_layer = TransformerDecoderLayer(256, 8, 2048, 0.1, "relu", False)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)

        # ! 정의된 decoder layer 구조를 바탕으로 decoder 정의
        # ! self.decoder = TransformerDecoder(decoder_layer, 6, nn.LayerNorm(256), True)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # ! Xavier initialization
        self._reset_parameters()

        self.d_model = d_model # ! 256
        self.nhead = nhead # ! 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):

        # ! src         = (B, 256, H, W) -> embedding된 feature maps
        # ! mask        = (B, H, W)      -> zero-padding masks
        # ! query_embed = (100, 256)     -> 정의된 trainable object queries
        # ! pos_embed   = (B, 256, H, W) -> 계산된 positional encodings

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        # ! Embedding된 feature maps를 1D로 flatten함
        src = src.flatten(2).permute(2, 0, 1)
        # ! src.flatten(2) = (B, 256, HW)
        # ! src.flatten(2).permute(2, 0, 1) = (HW, B, 256)

        # ! 마찬가지로 positional encodings도 1D로 flatten함
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # ! (HW, B, 256)

        # ! Object queries를 batch size에 맞게 padding함
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) # ! (100, B, 256)

        # ! Masks도 1D로 flatten함
        mask = mask.flatten(1) # ! (B, HW)

        # ! 결과를 저장할 tensor 선언
        tgt = torch.zeros_like(query_embed) # ! (100, B, 256)

        # ! Transformer encoder에 입력
        # ! src       = (HW, B, 256)
        # ! mask      = (B, HW)
        # ! pos_embed = (HW, B, 256)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # ! memory = (HW, B, 256) -> 6개의 Transformer encoder layers를 통과한 output

        # ! Encoder output을 decoder로 입력함
        # ! tgt         = (100, B, 256) -> 결과를 저장할 tensor
        # ! memory      = (HW, B, 256)  -> 6개의 Transformer encoder layers를 통과한 output
        # ! mask        = (B, HW)       -> 1D로 flatten된 masks
        # ! pos_embed   = (HW, B, 256)  -> 1D로 flatten된 positional encodings
        # ! query_embed = (100, B, 256) -> object queries
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # ! hs = (6, 100, B, 256) -> 각 decoder layer의 output = (100, B, 256)이 순서대로 총 6개 stack되어 있음

        # ! hs.transpose(1, 2)                        = (6, B, 100, 256) -> 각 decoder layer의 output = (B, 100, 256)이 순서대로 총 6개 stack되어 있음
        # ! memory.permute(1, 2, 0).view(bs, c, h, w) = (B, 256, H, W)   -> 6개의 encoder layers를 통과한 최종 Transformer encoder output

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()

        # ! 정의된 encoder layer 구조를 바탕으로 6개의 encoder layer로 구성된 Transformer encoder 정의
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers # ! 6
        self.norm = norm # ! None

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        # ! src                  = (HW, B, 256)
        # ! mask                 = None
        # ! src_key_padding_mask = (B, HW)
        # ! pos_embed            = (HW, B, 256)

        output = src
        
        # ! 총 6개의 encoder layer를 통과시킴
        for layer in self.layers:

            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            # ! output = (HW, B, 256) -> 하나의 Transformer encoder layer를 통과한 output

        # ! Pre-normalization을 하지 않음
        # ! self.norm = None
        if self.norm is not None:
            output = self.norm(output)

        # ! output = (HW, B, 256) -> 6개의 Transformer encoder layers를 통과한 output

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()

        # ! 정의된 decoder layer 구조를 바탕으로 6개의 decoder layer로 구성된 Transformer decoder 정의
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers # ! 6
        self.norm = norm # ! nn.LayerNorm(256)
        self.return_intermediate = return_intermediate # ! True

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # ! tgt                     = (100, B, 256) -> 결과를 저장할 tensor
        # ! memory                  = (HW, B, 256)  -> 6개의 Transformer encoder layers를 통과한 output
        # ! tgt_mask                = None
        # ! memory_mask             = None
        # ! tgt_key_padding_mask    = None
        # ! memory_key_padding_mask = (B, HW)       -> 1D로 flatten된 masks
        # ! pos                     = (HW, B, 256)  -> 1D로 flatten된 positional encodings
        # ! query_pos               = (100, B, 256) -> object queries

        output = tgt # ! (100, B, 256)

        # ! 각 intermediate decoders의 output을 저장하기 위한 리스트 선언
        intermediate = []

        # ! 6개의 decoder layer 통과
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # ! output = (100, B, 256) -> 하나의 Transformer decoder layer를 통과한 output

            # ! 모든 decoder layers의 output을 저장함
            if self.return_intermediate: # ! True
                intermediate.append(self.norm(output))

        # ! self.norm = nn.LayerNorm(256)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # ! True
        if self.return_intermediate:
            return torch.stack(intermediate)

        # ! torch.stack(intermediate) = (6, 100, B, 256) -> 각 decoder layer의 output = (100, B, 256)이 순서대로 총 6개 stack되어 있음

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # ! Self-attention layer 정의
        # ! self.self_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        # ! FFN layer 정의
        # ! self.linear1 = nn.Linear(256, 2048)
        # ! self.dropout = nn.Dropout(0.1)
        # ! self.linear2 = nn.Linear(2048, 256)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # ! Normalization 및 dropout 정의
        # ! self.norm1 = nn.LayerNorm(256)
        # ! self.norm2 = nn.LayerNorm(256)
        # ! self.dropout1 = nn.Dropout(0.1)
        # ! self.dropout2 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation) # ! ReLU
        self.normalize_before = normalize_before # ! False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        # ! src                  = (HW, B, 256)
        # ! src_mask             = None
        # ! src_key_padding_mask = (B, HW)
        # ! pos                  = (HW, B, 256)

        # ! Positional encoding 반영하고 self-attention 수행
        # ! Q = q                    = (HW, B, 256)
        # ! K = k                    = (HW, B, 256)
        # ! V = src                  = (HW, B, 256)
        # ! src_mask                 = None
        # ! src_key_padding_mask     = (B, HW)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # ! nn.MultiheadAttention의 output은 self-attention output, attention weights로 구성된 tuple임
        # ! src2는 self-attention output만을 선택함 = (HW, B, 256)
        
        # ! Residual connection + dropout, normalization 적용 
        src = src + self.dropout1(src2) # ! (HW, B, 256)
        src = self.norm1(src) # ! (HW, B, 256)

        # ! FFN 수행
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # ! (HW, B, 256)

        # ! Residual connection + dropout, normalization 적용  
        src = src + self.dropout2(src2) # ! (HW, B, 256)
        src = self.norm2(src) # ! (HW, B, 256)

        # ! src = (HW, B, 256) -> 하나의 Transformer encoder layer를 통과한 output

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        # ! src                  = (HW, B, 256)
        # ! src_mask             = None
        # ! src_key_padding_mask = (B, HW)
        # ! pos                  = (HW, B, 256)

        # ! Pre-normalization 하지 않음
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        
        # ! output 계산 
        # ! self.forward_post(src, src_mask, src_key_padding_mask, pos) = (HW, B, 256) -> 한 번의 Transformer encoder layer를 통과한 output

        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # ! Decoder self-attention layer 정의
        # ! Decoder는 하나의 decoder layer에서 모든 N개의 objects를 예측하므로 encoder self-attention과 같은 방식으로 정의됨 (masking X)
        # ! self.self_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # ! Encoder-decoder attention 정의
        # ! Q = decoder의 queries, K, V = encoder의 key, values를 사용함
        # ! self.multihead_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        # ! FFN layer 정의
        # ! self.linear1 = nn.Linear(256, 2048)
        # ! self.dropout = nn.Dropout(0.1)
        # ! self.linear2 = nn.Linear(2048, 256)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # ! Normalization 및 dropout 정의
        # ! self.norm1 = nn.LayerNorm(256)
        # ! self.norm2 = nn.LayerNorm(256)
        # ! self.norm3 = nn.LayerNorm(256)
        # ! self.dropout1 = nn.Dropout(0.1)
        # ! self.dropout2 = nn.Dropout(0.1)
        # ! self.dropout3 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation) # ! ReLU
        self.normalize_before = normalize_before # ! False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        
        # ! tgt                     = (100, B, 256) -> 결과를 저장할 tensor
        # ! memory                  = (HW, B, 256)  -> 6개의 Transformer encoder layers를 통과한 output
        # ! tgt_mask                = None
        # ! memory_mask             = None
        # ! tgt_key_padding_mask    = None
        # ! memory_key_padding_mask = (B, HW)       -> 1D로 flatten된 masks
        # ! pos                     = (HW, B, 256)  -> 1D로 flatten된 positional encodings
        # ! query_pos               = (100, B, 256) -> object queries

        # ! Positional encoding 반영하고 decoder self-attention 수행
        # ! Q = q                    = (100, B, 256)
        # ! K = k                    = (100, B, 256)
        # ! V = tgt                  = (100, B, 256)
        # ! tgt_mask                 = None
        # ! tgt_key_padding_mask     = None
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ! nn.MultiheadAttention의 output은 self-attention output, attention weights로 구성된 tuple임
        # ! tgt2는 self-attention output만을 선택함 = (100, B, 256)

        # ! Residual connection + dropout, normalization 적용 
        tgt = tgt + self.dropout1(tgt2) # ! (100, B, 256)
        tgt = self.norm1(tgt) # ! (100, B, 256)

        # ! Positional encoding 반영하고 encoder-decoder attention 적용
        # ! Q = self.with_pos_embed(memory, pos)    = (100, B, 256)
        # ! K = self.with_pos_embed(tgt, query_pos) = (HW, B, 256)
        # ! V = memory                              = (HW, B, 256)
        # ! memory_mask                             = None
        # ! memory_key_padding_mask                 = (B, HW)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # ! nn.MultiheadAttention의 output은 self-attention output, attention weights로 구성된 tuple임
        # ! tgt2는 self-attention output만을 선택함 = (100, B, 256)
        
        # ! Residual connection + dropout, normalization 적용 
        tgt = tgt + self.dropout2(tgt2) # ! (100, B, 256)
        tgt = self.norm2(tgt) # ! (100, B, 256)

        # ! FFN 수행
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # ! (100, B, 256)

        # ! Residual connection + dropout, normalization 적용
        tgt = tgt + self.dropout3(tgt2) # ! (100, B, 256)
        tgt = self.norm3(tgt) # ! (100, B, 256)

        # ! tgt = (100, B, 256) -> 하나의 Transformer decoder layer를 통과한 output 

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # ! tgt                     = (100, B, 256) -> 결과를 저장할 tensor
        # ! memory                  = (HW, B, 256)  -> 6개의 Transformer encoder layers를 통과한 output
        # ! tgt_mask                = None
        # ! memory_mask             = None
        # ! tgt_key_padding_mask    = None
        # ! memory_key_padding_mask = (B, HW)       -> 1D로 flatten된 masks
        # ! pos                     = (HW, B, 256)  -> 1D로 flatten된 positional encodings
        # ! query_pos               = (100, B, 256) -> object queries

        # ! Pre-normalization 하지 않음 
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        
        # ! output 계산 
        # ! self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos) 
        # ! = (100, B, 256) -> 한 번의 Transformer decoder layer를 통과한 output

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):

    # ! Transformer 모델 생성
    # ! args.hidden_dim      = 256
    # ! args.dropout         = 0.1
    # ! args.nheads          = 8
    # ! args.dim_feedforward = 2048
    # ! args.enc_layers      = 6
    # ! args.dec_layers      = 6
    # ! args.pre_norm        = False
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
