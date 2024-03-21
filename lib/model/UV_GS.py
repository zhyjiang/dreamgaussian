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
from lib.model.UV_encoder import UV_Transformer
from lib.data_utils.uv_map_generator import UV_Map_Generator






class UV_GS(nn.Module):

    def __init__(self, opt):
        super(UV_GS,self).__init__()
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator = UV_Map_Generator(
            UV_height=224,
            UV_pickle='template.pickle'
        )
        self.uv_model = None
        if self.opt.input_mode != 'image+smpl':
            self.opt.input_mode = 'image'
            self.uv_model = UV_Transformer(self.opt,generator=self.generator).to(self.device)
        self.opt.input_mode = 'image+smpl'
        self.model = UV_Transformer(self.opt,generator=self.generator).to(self.device)

    def forward(self, pose,shape,x,cam=None):
        shs=None
        features_shs = None
        if self.uv_model is not None:
            shs,features_shs = self.uv_model(x,img=x,cam = None)
        
        means3D, opacity, scales,shs_2, rotations,features_all = self.model((pose,shape),img=x,cam = None)
        if shs is None:
            shs = shs_2
        if features_shs is None:
            features_shs = features_all
        return means3D, opacity, scales,shs, rotations,features_shs,features_all