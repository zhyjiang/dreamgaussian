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
import torch.nn as nn
import torch
import numpy as np

from transformers import ViTImageProcessor, ViTModel, ViTConfig
from lib.model.triplane import TriplaneLearnablePositionalEmbedding
from PIL import Image
import requests

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .utils import get_1d_sincos_pos_embed



class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
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
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # import ipdb;ipdb.set_trace()
        
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        # if tgt.size(0) > 10000 or tgt.size(1) > 10000:/
            # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # try:
        # import ipdb;ipdb.set_trace()
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        # except:
        #     import ipdb;ipdb.set_trace()
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
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
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
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








class VertexTransformer2(nn.Module):
    def __init__(self, opt,hidden_dim=64, num_joints=6890, num_layers=2, pose_dim=3, nhead=4, dropout=0.1,
                 dim_head=64    , mlp_dim=64,downsample_dim = 196,img_dim=4096,pose_num=24,H=1024,W=1024,device='cuda'):
        super().__init__()
        
        self.opt = opt
        
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.camera_param = self.opt.camera_param
        self.pose_dim = pose_dim
        self.device = device
        self.downsample_dim = downsample_dim
        self.dino = self.opt.dino 
        self.param_input = self.opt.param_input
        self.cross_attention=self.opt.cross_attn
        self.multi_view = self.opt.multi_view
        self.dino_update = self.opt.dino_update and  self.opt.dino
        self.upsample = self.opt.upsample
        self.ln_scale_weight = nn.Parameter( torch.randn(num_joints*self.upsample,1)*0.05,requires_grad=True)
        
        # if self.opt.trans_decoder:
        # import ipdb;ipdb.set_trace()
        
        if self.opt.dino:
            if self.cross_attention:

                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.downsample_dim, hidden_dim)),requires_grad=True)
                self.upsample_conv = nn.Conv1d(downsample_dim, num_joints*self.upsample, kernel_size=1)  
            else:
                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.downsample_dim+img_dim, hidden_dim)),requires_grad=True)
                self.upsample_conv = nn.Conv1d(downsample_dim+img_dim, num_joints*self.upsample, kernel_size=1) 
        else:
            self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.downsample_dim, hidden_dim)),requires_grad=True)
            self.upsample_conv = nn.Conv1d(self.downsample_dim, num_joints*self.upsample, kernel_size=1) 
        # self.cls_token = nn.Parameter(torch.randn((hidden_dim)))
        
        self.proj_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.pre_emb = nn.Linear(pose_dim, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        if self.param_input:
            self.pre_conv = nn.Conv1d(2,self.downsample_dim, kernel_size=1)
        else:
            self.pre_conv = nn.Conv1d(num_joints,self.downsample_dim, kernel_size=1)
        
        # self.downsamnple_conv = nn.Conv1d(img_dim, img_dim//2, kernel_size=1)
        
        if self.param_input:
            self.beta = nn.Linear(10, hidden_dim, bias=False)
            
            self.theta = nn.Linear(pose_num*3, hidden_dim, bias=False)
            self.rotate = nn.Linear(3, hidden_dim, bias=False)
            self.trans = nn.Linear(3, hidden_dim, bias=False)
        if self.dino:
            if self.opt.full_token and self.opt.reshape: 
                self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
                config=ViTConfig.from_pretrained('facebook/dino-vits16')
                self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16',config=config,ignore_mismatched_sizes=True).to(self.device)
            elif self.opt.full_token:
                self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
                self.processor.size['height'] = H
                self.processor.size['width'] = W
                
                config=ViTConfig.from_pretrained('facebook/dino-vits16')
                config.image_size = (H, W)
                self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16',config=config,ignore_mismatched_sizes=True).to(self.device)
            else:
                self.dino_encoder = torch.hub.load('facebookresearch/dino:main', self.dino.path).patch_embed.to(self.device)

        # if self.dino_update:
        #     self.dino_encoder.train()
        # else:
        #     self.dino_encoder.eval()

        # self.encoder = Transformer(hidden_dim, num_layers, nhead, dim_head, mlp_dim, cross=self.cross_attention,dropout=dropout)
        
        encoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                dropout)
        self.encoder = TransformerDecoder(encoder_layer, num_layers, None,
                                            return_intermediate=False)
        # decoder_norm = nn.LayerNorm(hidden_dim)
        
        # self.encoder = TransformerEncoder(encoder_layer, num_layers)
        if self.opt.triplane:
            self.tokenizer = TriplaneLearnablePositionalEmbedding(plane_size=64, num_channels=hidden_dim)
            backbone_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                dropout)
            self.backbone = TransformerDecoder(encoder_layer, num_layers, None,
                                            return_intermediate=False)
            
            # self.post_processor = TriplaneLearnablePositionalEmbedding(plane_size=32, num_channels=hidden_dim)
            
        
        if self.opt.trans_decoder:
            # self.decoder = Transformer(hidden_dim, num_layers, nhead, dim_head, mlp_dim, cross=self.cross_attention,dropout=dropout)
            decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                dropout)
            decoder_norm = nn.LayerNorm(hidden_dim)
            self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm,
                                            return_intermediate=False)
            
            
        self.dropout = nn.Dropout(dropout)
        
        self.merge_views = nn.Conv1d(self.opt.multi_view, 1, kernel_size=1) if self.opt.multi_view>1 else None
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.upsample = self.opt.upsample
        self.cam_proj = nn.Linear(4*4, 384)
        self.img_down = nn.Linear(384,hidden_dim)
        if self.opt.trans_decoder:
            self.learning_query = nn.Parameter(torch.randn((self.opt.batch_size, num_joints*self.upsample,hidden_dim)), requires_grad=True)
            self.decoder_pos_embed = nn.Parameter(torch.randn((self.opt.batch_size, num_joints*self.upsample, hidden_dim)), requires_grad=True)
            # self.upsample_layer = nn.Conv1d(num_joints*self.upsample//4, num_joints*self.opt.upsample, kernel_size=1)
        
        self.initialize_weights()

        # self.temp_head = self._make_head(hidden_dim,14)
        # import ipdb;ipdb.set_trace()
        self.mean3D_head = self._make_head(hidden_dim, 3)
        self.opacity_head = self._make_head(hidden_dim, 1)
        self.shs_head = self._make_head(hidden_dim, 3)
        self.rotations_head = self._make_head(hidden_dim, 4)
        # self.scale_vector = torch.tensor([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]).to(self.device)
        # self.scales_head = self._make_head(hidden_dim, 21)
        self.scales_head = self._make_head(hidden_dim, 3)

        
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def _make_head(self, hidden_dim, out_dim):
        layers = nn.Sequential(nn.Linear(self.hidden_dim, hidden_dim),
                               nn.GELU(),
                               nn.Dropout(0.1),
                               nn.Linear(hidden_dim, out_dim))
        return layers    
    
    def initialize_weights(self):

        pos_embed = get_1d_sincos_pos_embed(self.positional_emb.shape[-1], np.arange(self.positional_emb.shape[1]))
        self.positional_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, img=None,cam=None,mask_ratio=0.0):
        # import ipdb;ipdb.set_trace()

        mask = None
        # bs = x.shape[0]
       
        if self.dino:
            # import ipdb;ipdb.set_trace()

            if not self.dino_update and self.opt.full_token:
                with torch.no_grad():
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                    img_emb = self.dino_encoder(**inputs)
                    img_emb = img_emb['last_hidden_state'][...,1:,:]
                    
            elif self.opt.full_token:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                img_emb = self.dino_encoder(**inputs)
                img_emb = img_emb['last_hidden_state'][...,1:,:]
            elif self.dino_update:
                img_emb = self.dino_encoder(img)
            else:
                with torch.no_grad():
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)
                    img_emb = self.dino_encoder(img)

           
            
            if self.camera_param:
                assert cam is not None
                cam_emb = self.cam_proj(cam.reshape(cam.shape[0],-1))[:,None,:]
                emb = self.img_down(img_emb+ cam_emb)
            else:
                emb = self.img_down(img_emb[0].unsqueeze(0))
            
        if self.param_input:
            theta = x[0]
            beta = x[1]
          
            theta = self.theta(theta)
            beta = self.beta(beta)
            
            if len(theta.shape) == 2:
                theta = theta[:,None,:]
                beta = beta[:,None,:]
                
            x = torch.cat((theta, beta),dim=1)
            x = self.pre_norm(x)
            if mask is not None:
                x = self.masking(x, mask)
            # import ipdb;ipdb.set_trace()
            x = self.pre_conv(x)
            
        else:
            if x.shape[-1] != self.pose_dim:
                mask = x[:, :, -1] == 0
                x = x[:, :, :-1]
            x = self.pre_emb(x)
            x = self.pre_norm(x)
        
            if mask is not None:
                x = self.masking(x, mask)
            x = self.pre_conv(x)
        if self.dino and not self.cross_attention:

            x = torch.cat((emb, x), dim=1)
            x = self.pre_conv(x)  # 6890+4096 -> 2000

        # x = x + self.positional_emb
        # x = self.dropout(x)
        
        # import ipdb;ipdb.set_trace()
        x = x.permute(1,0,2)
        # import ipdb;ipdb.set_trace()
        memory = self.encoder(x, emb.permute(1,0,2),memory_key_padding_mask=mask, pos=self.positional_emb.permute(1,0,2),query_pos=None)
        memory = memory.squeeze(0)
        
        if self.opt.triplane:
            # import ipdb;ipdb.set_trace()
            tokens = self.tokenizer(x.shape[1], cond_embeddings=memory)
            
            tokens = self.backbone(
                    tokens,
                    encoder_hidden_states=emb,
                    modulation_cond=None,
                )

            scene_codes = self.tokenizer.detokenize(tokens)
            
            import ipdb;ipdb.set_trace()

            # import ipdb;ipdb.set_trace()
            # self.learning_query = self.learning_query.permute(1,0,2)
            final_features = torch.cat((memory,scene_codes,emb),dim=1)
            
            hs = self.decoder(self.learning_query.permute(1,0,2), final_features, memory_key_padding_mask=mask,
                          pos= self.positional_emb.permute(1,0,2),query_pos=self.decoder_pos_embed.permute(1,0,2))
        else:
            hs = self.decoder(self.learning_query.permute(1,0,2), memory, memory_key_padding_mask=mask,
                            pos= self.positional_emb.permute(1,0,2),query_pos=self.decoder_pos_embed.permute(1,0,2))
            
        # import ipdb;ipdb.set_trace()
        x = hs.squeeze(0).permute(1,0,2)
        # x = self.upsample_layer(x)
        
        
        # if self.cross_attention:
        #     # import ipdb;ipdb.set_trace()
        #     x = self.encoder(x,context=emb)
        # else:
        #     x = self.encoder(x)

        # import ipdb;ipdb.set_trace()
        # if self.multi_view>1:
            
        #     x = torch.transpose(x, 0, 1)
        #     x = self.merge_views(x)
            
        #     x = torch.transpose(x, 0, 1)
        
        
        # import ipdb;ipdb.set_trace()
        # if self.opt.trans_decoder:
        #     x = self.decoder((self.learning_query+self.decoder_pos_embed),context=x)
        #     # x = self.upsample_layer(x)
        # if not self.opt.trans_decoder:
        #     x = self.upsample_conv(x) # 2000+4096 -> 6890*n
        # res = self.temp_head(x)
        
        # means3D, opacity, scales, shs, rotations = res[...,0:3] , res[...,3:4],res[...,4:7],res[...,7:11],res[...,11:14]
        # opacity = opacity.sigmoid()
        # rotations = rotations.sigmoid()
        # scales = scales.sigmoid()
        # import ipdb;ipdb.set_trace()
        
        means3D = self.mean3D_head(x)
        opacity = self.opacity_head(x).sigmoid()
        shs = self.shs_head(x)
        # rotations = torch.nn.functional.normalize(self.rotations_head(x))
        rotations = self.rotations_head(x).sigmoid()
        scales = self.scales_head(x).sigmoid()
        # import ipdb;ipdb.set_trace()

        # import ipdb;ipdb.set_trace()

        # scales = self.softmax(self.scales_head(x).reshape(x.shape[0],-1,3,7))
        # scales = torch.einsum('ijlk,k->ijl', scales, self.scale_vector)
        # import ipdb;ipdb.set_trace()

        return means3D, opacity, scales, shs, rotations
    
    def masking(self, x, mask):
        x[mask] = self.mask_token
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        x[mask] = self.mask_token

        return x, mask, ids_restore
