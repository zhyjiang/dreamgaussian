# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# DETR Transformer class.

# Copy-paste from torch.nn.Transformer with modifications:
#     * positional encodings are passed in MHattention
#     * extra LN at the end of encoder is removed
#     * decoder returns a stack of activations from all decoding layers
# """
# import copy
# from typing import Optional, List

# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# import torch.nn as nn
# import torch
# import numpy as np
# from lib.model.UV_encoder import UV_Transformer
# from lib.data_utils.uv_map_generator import UV_Map_Generator


# class UV_GS(nn.Module):

#     def __init__(self, opt):
#         super(UV_GS,self).__init__()
#         self.opt = opt
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.generator = UV_Map_Generator(
#             UV_height=224,
#             UV_pickle='template.pickle'
#         )

#         if self.opt.double_vertex:
#             import ipdb; ipdb.set_trace()
#             vt_faces = self.generator.vt_faces.to(self.device)
#             vt_faces_to_v = vt_faces[self.generator.v_to_vt] 
#             midpoint1 = (vt_faces_to_v[:,:,0:1] + vt_faces_to_v[:,:,1:2]) / 2
#             midpoint2 = (vt_faces_to_v[:,:, 1:2] + vt_faces_to_v[:,:,2:]) / 2
#             midpoint3 = (vt_faces_to_v[:,:,2:] + vt_faces_to_v[:,:, 0:1]) / 2 # 13776 * 3 *6890
#             self.generator.midpoints_to_v = torch.concatenate([ midpoint1, midpoint2, midpoint3], axis=2) # 13776 * 3 * 6890 * 3
        
#         self.uv_model = None
#         if self.opt.input_mode != 'image+smpl':
#             self.opt.input_mode = 'image'
#             self.uv_model = UV_Transformer(self.opt,generator=self.generator).to(self.device)
#         self.opt.input_mode = 'image+smpl'
#         self.model = UV_Transformer(self.opt,generator=self.generator).to(self.device)

#     def forward(self, pose,shape,x,cam=None):
#         shs=None
#         features_shs = None
#         if self.uv_model is not None:
#             shs,features_shs = self.uv_model(x,img=x,cam = None)
        
#         means3D, opacity, scales,shs_2, rotations,features_all = self.model((pose,shape),img=x,cam = None)
#         if shs is None:
#             shs = shs_2
#         if features_shs is None:
#             features_shs = features_all
#         return means3D, opacity, scales, shs, rotations, features_shs, features_all



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
            UV_height=224 if self.opt.uv_map_224 else 448,
            UV_pickle='template.pickle'
        )

        self.uv_model = UV_Transformer(self.opt,generator=self.generator,uv=True,hidden_dim=768,dim_head=768).to(self.device)
        if self.opt.pretrain_uv:
            self.shape_model = None
        else:
            self.shape_model = UV_Transformer(self.opt,generator=self.generator,shape=True,hidden_dim=128,dim_head=128).to(self.device)

        
        if self.opt.triple_point:
            vt_faces = torch.tensor(self.generator.vt_faces).to(self.device)
            vt_faces_to_v = self.generator.v_to_vt[vt_faces]
            midpoint1 = (vt_faces_to_v[:,0:1,:] + vt_faces_to_v[:,1:2,:]) / 2
            midpoint2 = (vt_faces_to_v[:, 1:2,:] + vt_faces_to_v[:,2:,:]) / 2
            midpoint3 = (vt_faces_to_v[:,2:,:] + vt_faces_to_v[:,0:1,:]) / 2 # 13776 * 3 *6890
            self.generator.midpoints_to_v = midpoint1 # 13776 * 3 * 6890
            
 
    def freeze(self,model):
        for param in model.parameters():
            param.requires_grad = False
            
    def activate(self,model):
        for param in model.parameters():
            param.requires_grad = True
            
    def check_no_grad(self,model):
        for param in model.parameters():
            if param.requires_grad:
                import ipdb;ipdb.set_trace()
            # param.requires_grad = False
            
    def check_grad(self,model):
        for param in model.parameters():
            if not param.requires_grad:
                import ipdb;ipdb.set_trace()

    def forward(self, pose,shape,x,cam=None,pretrain_uv=False):

        if pretrain_uv:
          
            img_token,shs =  self.uv_model((pose,shape),img=x,cam = None)
            return shs
        else:
            if self.opt.train_strategy == 'uv':
                # self.freeze(self.shape_model)
                # self.check_no_grad(self.shape_model)
               
                # self.uv_model.train()
                
                # self.check_grad(self.uv_model)
                img_token,shs = self.uv_model((pose,shape),img=x,cam = None)
                
                with torch.no_grad():

                    means3D, opacity, scales, rotations = self.shape_model((pose,shape),img=img_token,cam = None)
                return means3D, opacity, scales,shs, rotations
            elif self.opt.train_strategy == 'shape':
                # self.freeze(self.uv_model)
                # self.check_no_grad(self.uv_model)
                
                with torch.no_grad():
                    img_token,shs = self.uv_model((pose,shape),img=x,cam = None)

                # self.shape_model.train()
                # self.check_grad(self.shape_model)
                means3D, opacity, scales, rotations = self.shape_model((pose,shape),img=img_token,cam = None)
                return means3D, opacity, scales,shs, rotations
            elif self.opt.train_strategy == 'both':
                img_token,shs = self.uv_model((pose,shape),img=x,cam = None)
                
                
                means3D, opacity,scales, rotations = self.shape_model((pose,shape),img=img_token,cam = None)

                return means3D, opacity, scales,shs, rotations
                
                


        
        # shs=None
        # features_shs = None
        # if self.uv_model is not None:
        #     shs,features_shs = self.uv_model(x,img=x,cam = None)
        
        # means3D, opacity, scales,shs, rotations,features= self.model((pose,shape),img=x,cam = None)
        # if shs is None:
        #     shs = shs_2
        # if features_shs is None:
        #     features_shs = features_all
        # return means3D, opacity, scales,shs, rotations,features