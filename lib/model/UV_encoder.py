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

from lib.data_utils.uv_map_generator import UV_Map_Generator

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

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

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
            # import ipdb;ipdb.set_trace()
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
        
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
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



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SuperresolutionHybrid(nn.Module):
    def __init__(self):
        super(SuperresolutionHybrid,self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # Further upsampling
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final upsampling to target size
        self.upsample3 = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x[None,...]
        # import ipdb; ipdb.set_trace()
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        # x = self.upsample2(x)
        # x = F.relu(self.conv2(x))
        x = self.upsample3(x)
        x = self.conv3(x)
        return x



class UV_Transformer(nn.Module):
    def __init__(self, opt,generator=None,hidden_dim=768, num_joints=6890, num_layers=2, pose_dim=3, nhead=4, dropout=0.1,
                 dim_head=768,uv=False,shape=False,mlp_dim=64,downsample_dim = 196,img_dim=4096,pose_num=24,H=1024,W=1024,device='cuda'):
        super().__init__()
        
        self.opt = opt
        
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.camera_param = self.opt.camera_param
        self.pose_dim = pose_dim
        self.device = device
        self.downsample_dim = downsample_dim
        self.input_mode = self.opt.input_mode

        self.dino = self.opt.dino and self.input_mode!='smpl'
        self.param_input = self.opt.param_input
        self.cross_attention=self.opt.cross_attn
        self.multi_view = self.opt.multi_view
        self.dino_update = self.opt.dino_update and  self.opt.dino
        self.upsample = self.opt.upsample
        if self.opt.learned_scale:
            self.ln_scale_weight = nn.Parameter( torch.randn(num_joints*self.upsample,1)*0.05,requires_grad=True)
        self.img_channel= self.opt.dataset.img_channel
        self.mode = 'uv' if uv else 'shape'

        # self.w = torch.nn.Parameter(torch.randn(6890*2,6890),requires_grad=True)
        
        
        # if self.input_mode == 'image+smpl':
        #     self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.img_channel+2, hidden_dim)),requires_grad=True)
        # else:
        #     self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.img_channel, hidden_dim)),requires_grad=True)
        if uv:
            self.positional_emb_q = nn.Parameter(torch.randn((self.multi_view,28*28, hidden_dim)),requires_grad=True)
            self.learned_uv_q = nn.Parameter(torch.randn((self.multi_view,28*28, hidden_dim)),requires_grad=True)
            if self.opt.dino_interpolation:
                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, 4096, hidden_dim)),requires_grad=True)
            else:
                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.img_channel, hidden_dim)),requires_grad=True)
        else:
            self.positional_emb_q = nn.Parameter(torch.randn((self.multi_view,6890, hidden_dim)),requires_grad=True)
            self.learned_uv_q = nn.Parameter(torch.randn((self.multi_view,6890, hidden_dim)),requires_grad=True)
            if self.opt.dino_interpolation:
                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, 4098, hidden_dim)),requires_grad=True)
            else:
                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.img_channel+2, hidden_dim)),requires_grad=True)
            
        # if self.opt.uv_query:
        #     self.learned_down = nn.Sequential(nn.Linear(224*224, 56*56),
        #                         nn.GELU(),
        #                         nn.Dropout(0.1),
        #                         nn.Linear(56*56, 56*56))
        self.generator = generator
        if self.param_input and not uv:
            self.beta = nn.Linear(10, hidden_dim, bias=False)
            
            self.theta = nn.Linear(pose_num*3, hidden_dim, bias=False)
        if self.dino and uv:
            if self.opt.full_token and self.opt.reshape: 
                self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
                config=ViTConfig.from_pretrained('facebook/dino-vitb16')
                self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vitb16',config=config,ignore_mismatched_sizes=True).to(self.device)
            elif self.opt.full_token:
                self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
                self.processor.size['height'] = H
                self.processor.size['width'] = W
                
                config=ViTConfig.from_pretrained('facebook/dino-vitb16')
                config.image_size = (H, W)
                self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vitb16',config=config,ignore_mismatched_sizes=True).to(self.device)
            else:
                self.dino_encoder = torch.hub.load('facebookresearch/dino:main', self.dino.path).patch_embed.to(self.device)

        
        if self.opt.uv_query:
            if uv:
                encoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                    dropout)
                encoder_norm = nn.LayerNorm(hidden_dim)
                self.encoder = TransformerDecoder(encoder_layer, num_layers, encoder_norm,
                                                return_intermediate=False)
            else: #shape
                encoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                    dropout)
                encoder_norm = nn.LayerNorm(hidden_dim)
                self.encoder = TransformerDecoder(encoder_layer, num_layers, encoder_norm,
                                                return_intermediate=False)
        else:
            encoder_layer =  TransformerEncoderLayer(hidden_dim, nhead, dim_head,
                                                dropout)
            self.encoder = TransformerEncoder(encoder_layer, num_layers, None)

        if self.opt.triplane:
            self.tokenizer = TriplaneLearnablePositionalEmbedding(plane_size=64, num_channels=hidden_dim)
            backbone_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                dropout)
            self.backbone = TransformerDecoder(encoder_layer, num_layers, None,
                                            return_intermediate=False)
            
        
        if self.opt.trans_decoder:
            decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_head,
                                                dropout)
            decoder_norm = nn.LayerNorm(hidden_dim)
            self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm,
                                            return_intermediate=False)
            
            
        self.dropout = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.upsample = self.opt.upsample

        
        if uv :
            if self.opt.uv_query:
                self.upsampler = nn.Sequential(nn.Linear(28*28, 28*28),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(28*28,14*14*16*16))
            else:
                self.upsampler = nn.Sequential(nn.Linear(self.img_channel, self.img_channel),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(self.img_channel, 224*224))
            
            self.resample = nn.Linear(224*224,6890)

            self.shs_head = self._make_head(hidden_dim, 3)
            self.img_down = None
        else:

            self.mean3D_head = self._make_head(hidden_dim, 3)
            self.opacity_head = self._make_head(hidden_dim, 1)
            self.shs_head = self._make_head(hidden_dim, 3)
            self.rotations_head = self._make_head(hidden_dim, 4)
            self.scales_head = self._make_head(hidden_dim, 3)
            self.img_down = nn.Linear(768,hidden_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.initialize_weights()

    
    def _make_head(self, hidden_dim, out_dim):
        layers = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.GELU(),
                               nn.Dropout(0.1),
                               nn.Linear(hidden_dim, out_dim))
        return layers    
    
    def initialize_weights(self):

        pos_embed = get_1d_sincos_pos_embed(self.positional_emb.shape[-1], np.arange(self.positional_emb.shape[1]))
        self.positional_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
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
        mask = None
        if self.mode == 'uv':
            if self.dino :
                if not self.opt.dino_update:
                    for param in self.dino_encoder.parameters():
                        param.requires_grad = False
                    self.dino_encoder.eval()
                # else:
                #     for param in self.dino_encoder.parameters():
                #         param.requires_grad = True
                # with torch.no_grad():
                if self.opt.dino_interpolation:
                    inputs = self.processor(images=img, return_tensors="pt",do_resize=False,do_rescale=False,do_normalize=True).to(self.device)
                    img_emb = self.dino_encoder(**inputs,interpolate_pos_encoding=self.opt.dino_interpolation)
                else:
                    if img.shape[-1]== 224:
                        inputs = self.processor(images=img, return_tensors="pt",do_resize=False,do_rescale=False,do_normalize=True).to(self.device)
                    else:
                        inputs = self.processor(images=img, return_tensors="pt",do_resize=True,do_rescale=False,do_normalize=True).to(self.device)
                    img_emb = self.dino_encoder(**inputs)
                img_emb = img_emb['last_hidden_state'][...,1:,:]

                if self.camera_param:
                    assert cam is not None
                    cam_emb = self.cam_proj(cam.reshape(cam.shape[0],-1))[:,None,:]
                    emb = self.img_down(img_emb+ cam_emb)
                else:
                    if self.img_down is not None:
                        emb = self.img_down(img_emb[0].unsqueeze(0))
                    else:
                        emb = img_emb
            else:
                assert self.opt.train_from_scratch is True
                emb = self.patch_embed(img.permute((0,3,1,2)))
            
            bs = img.shape[0] if img is not None else x.shape[0]
            img_token = emb.permute(1,0,2).clone()
            if self.opt.uv_query:
                x = emb.permute(1,0,2)
                x = self.encoder(self.learned_uv_q.permute(1,0,2),x,query_pos=self.positional_emb_q.permute(1,0,2),pos=self.positional_emb.permute(1,0,2))
                x = x.squeeze(0).permute(1,2,0)
                x = self.upsampler(x)
                x = x.view(224,224,-1)
                features = x.clone()


            else:

                x= self.encoder(x,src_key_padding_mask=mask, pos=self.positional_emb.permute(1,0,2))
                x = x.permute((1,2,0))
                x = self.upsampler(x)
                x = x.view(224,224,-1)
                features = x.clone()

            x = self.generator.resample(x).view((bs,-1,self.hidden_dim))
            shs = self.shs_head(x)
            return img_token,shs

        else:
           
            img = self.img_down(img)
            theta = x[0]
            beta = x[1]
            theta = self.theta(theta)
            beta = self.beta(beta)
            if len(theta.shape) == 2:
                theta = theta[:,None,:]
                beta = beta[:,None,:]
                
            x = torch.cat((theta, beta),dim=1)
            if mask is not None:
                x = self.masking(x, mask)
            x = torch.cat((img, x.permute(1,0,2)), dim=0)
          
            if self.opt.uv_query:
                x = self.encoder(self.learned_uv_q.permute(1,0,2),x,query_pos=self.positional_emb_q.permute(1,0,2),pos=self.positional_emb.permute(1,0,2))
                x = x.squeeze(0).permute((1,0,2))
                features = x.clone()


            else:

                x= self.encoder(x,src_key_padding_mask=mask, pos=self.positional_emb.permute(1,0,2))
                if self.input_mode != 'image':  
                    x = x[:-2,...]
                x = x.permute((1,2,0))
                x = self.upsampler(x)
                x = x.view(224,224,-1)
                features = x.clone()
        
            opacity = self.opacity_head(x).sigmoid()
            means3D = self.mean3D_head(x)
            rotations = F.normalize(self.rotations_head(x))
            
            if self.opt.exp_scale:
                scales = torch.exp(self.scales_head(x))
                scales  = torch.clamp(scales, min=0, max=self.opt.clip_scaling)
            else:
                scales = self.scales_head(x).sigmoid()
            
            if self.opt.return_RT:
                new_R = self.R_head(x).view((bs,-1,3,3))
                new_T = self.T_head(x).view((bs,-1,3))
                means3D = torch.matmul(means3D.unsqueeze(2),new_R).squeeze() + new_T
            return means3D, opacity, scales, rotations
        


    
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
    
    
    
