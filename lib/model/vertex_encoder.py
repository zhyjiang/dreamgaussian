import torch.nn as nn
import torch
import numpy as np

from transformers import ViTImageProcessor, ViTModel, ViTConfig
from PIL import Image
import requests

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .utils import get_1d_sincos_pos_embed


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # import ipdb;ipdb.set_trace()
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.query_qkv  = nn.Linear(dim, inner_dim, bias = False)
        self.key_qkv  = nn.Linear(dim, inner_dim, bias = False)
        self.value_qkv  = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q,k=None,v=None):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        assert k is not None
        assert v is not None
   
        q = self.query_qkv(q)
        k = self.key_qkv(k)
        v = self.value_qkv(v)
        # import ipdb;ipdb.set_trace()
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,k,v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, cross=False, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.cross = cross
        for _ in range(depth):
            if self.cross:
                self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                PreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                    
                ]))
            
            
                
    def forward(self, x, context=None):
        
        if context is not None  and len(context.shape) > 3:
            context = context.squeeze(0)
        
        if not self.cross:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        else:
             for attn,  attn2, ff in self.layers:
                    # import ipdb;ipdb.set_trace()
                    x = attn(x) + x
                    # x = ff(x) + x
 
                    x = attn2(x,k=context,v=context) + x
                    x = ff(x) + x
            
            # if context is not None:
                
            
       
        return x

class VertexTransformer(nn.Module):
    def __init__(self, opt,hidden_dim=64, num_joints=6890, num_layers=2, pose_dim=3, nhead=4, dropout=0.1,
                 dim_head=64, mlp_dim=64, camera_param=True,has_bbox=False,upsample=1,downsample_dim = 1024,dino=False,dino_update=False,img_dim=4096,param_input=False,cross_attention=False,pose_num=24,multi_view=1,device='cuda'):
        super().__init__()
        
        self.opt = opt
        
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.camera_param = camera_param
        self.pose_dim = pose_dim
        self.device = device
        self.downsample_dim = downsample_dim
        self.param_input = param_input
        self.cross_attention=cross_attention
        self.multi_view = multi_view
        self.dino_update = dino_update and  dino
        
        if dino:
            if self.cross_attention:

                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.downsample_dim, hidden_dim)),requires_grad=True)
                self.upsample_conv = nn.Conv1d(downsample_dim, num_joints*upsample, kernel_size=1) if upsample!=1 else None
            else:
                self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.downsample_dim+img_dim, hidden_dim)),requires_grad=True)
                self.upsample_conv = nn.Conv1d(downsample_dim+img_dim, num_joints*upsample, kernel_size=1) if upsample!=1 else None
        else:
            self.positional_emb = nn.Parameter(torch.randn((self.multi_view, self.downsample_dim, hidden_dim)),requires_grad=True)
            self.upsample_conv = nn.Conv1d(self.downsample_dim, num_joints*upsample, kernel_size=1) if upsample!=1 else None
        # self.cls_token = nn.Parameter(torch.randn((hidden_dim)))
        
    
        self.proj_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.pre_emb = nn.Linear(pose_dim, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        if self.param_input:
            self.pre_conv = nn.Conv1d(4,self.downsample_dim, kernel_size=1)
        else:
            self.pre_conv = nn.Conv1d(num_joints,self.downsample_dim, kernel_size=1)
        
        # self.downsamnple_conv = nn.Conv1d(img_dim, img_dim//2, kernel_size=1)
        
        if self.param_input:
            self.shape = nn.Linear(10, hidden_dim, bias=False)
            
            self.pose = nn.Linear(pose_num*3, hidden_dim, bias=False)
            self.rotate = nn.Linear(3, hidden_dim, bias=False)
            self.trans = nn.Linear(3, hidden_dim, bias=False)
        self.dino = dino
        if self.dino:
            
            if self.opt.full_token and self.opt.reshape: 
                self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
                # self.processor.size['height'] = 1080
                # self.processor.size['width'] = 1920
                
                config=ViTConfig.from_pretrained('facebook/dino-vits16')
                # config.image_size = (1080, 1920)
                self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16',config=config,ignore_mismatched_sizes=True).to(self.device)
            elif self.opt.full_token:
               
                self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
                self.processor.size['height'] = 1080
                self.processor.size['width'] = 1920
                
                config=ViTConfig.from_pretrained('facebook/dino-vits16')
                config.image_size = (1080, 1920)
                self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16',config=config,ignore_mismatched_sizes=True).to(self.device)
            else:
               
            # import ipdb;ipdb.set_trace()
            # self.dino_encoder.config.image_size = (1080, 1920)

           
                self.dino_encoder = torch.hub.load('facebookresearch/dino:main', self.dino.path).to(self.device)
            
            # import ipdb;ipdb.set_trace()
        self.encoder = Transformer(hidden_dim, num_layers, nhead, dim_head, mlp_dim, cross=self.cross_attention,dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.merge_views = nn.Conv1d(multi_view, 1, kernel_size=1) if multi_view>1 else None
        
        self.has_bbox = has_bbox
        if has_bbox:
            self.bbox_tokenize = nn.Linear(4, hidden_dim, bias=False)
            
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.upsample = upsample
        self.cam_proj = nn.Linear(4*4, 384)
        self.img_down = nn.Linear(384,hidden_dim)
        self.img_downconv = nn.Conv1d(4096,2048,kernel_size=1)
        self.initialize_weights()
        
        self.mean3D_head = self._make_head(hidden_dim, 3)
        self.opacity_head = self._make_head(hidden_dim, 1)
        self.shs_head = self._make_head(hidden_dim, 3)
        self.rotations_head = self._make_head(hidden_dim, 4)
        self.scales_head = self._make_head(hidden_dim, 3)
    
    def _make_head(self, hidden_dim, out_dim):
        layers = nn.Sequential(nn.Linear(self.hidden_dim, hidden_dim),
                               nn.GELU(),
                               nn.Dropout(0.1),
                               nn.Linear(hidden_dim, out_dim))
        return layers    
    
    def initialize_weights(self):
        # import ipdb;ipdb.set_trace()
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
        
        mask = None
       
        if self.dino:
            # import ipdb;ipdb.set_trace()
            if not self.dino_update and self.opt.full_token:
                # import ipdb;ipdb.set_trace()
                # self.dino_encoder.train()
                with torch.no_grad():
                    self.dino_encoder.eval()
                # import ipdb;ipdb.set_trace()
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                # # # import ipdb;ipdb.set_trace()
                    img_emb = self.dino_encoder(**inputs)
                # # # import ipdb;ipdb.set_trace()
                    # import ipdb;ipdb.set_trace()
                    img_emb = img_emb['last_hidden_state'][...,1:,:]
                    
            elif self.opt.full_token:
                
                self.dino_encoder.train()
                
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                # # # import ipdb;ipdb.set_trace()
                img_emb = self.dino_encoder(**inputs)
                # # # import ipdb;ipdb.set_trace()
                    # import ipdb;ipdb.set_trace()
                img_emb = img_emb['last_hidden_state'][...,1:,:]
                # img_emb = img_emb['last_hidden_state'][...,1:,:]
            elif self.dino_update:
                self.dino_encoder.train()
                img_emb = self.dino_encoder(img)
            else:
                self.dino_encoder.eval()
                img_emb = self.dino_encoder(img)
                    
                    # import ipdb;ipdb.set_trace()
                    
                    # img_emb  = self.img_downconv(img_emb)
                    
                
                # new_batch = {"pixel_values": img.to(img.device)} # Apply torchvision.transforms to read PIL image and convert to Tensor
                # with torch.no_grad():
                # import ipdb;ipdb.set_trace()
                # img_emb = self.dino_encoder(**inputs).last_hidden_state
                    
                
                # inputs = self.processor(images=img, return_tensors="pt")
                # import ipdb;ipdb.set_trace()
                # outputs = self.dino_encoder(**inputs)
                # img_emb = outputs.last_hidden_state
                
                 # B*384
                # import ipdb;ipdb.set_trace()
                
                
                    # import ipdb;ipdb.set_trace()
                    
                    
            # else:
            #     self.dino_encoder.train()
            #     img_emb = self.dino_encoder(img)
            
            # import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()

            assert cam is not None
            cam_emb = self.cam_proj(cam.reshape(cam.shape[0],-1))[:,None,:]
            
            if self.camera_param:
                emb = self.img_down(img_emb+ cam_emb)
            else:
                # emb = self.img_down(img_emb.unsqueeze(0))
                # import ipdb;ipdb.set_trace()
                emb = self.img_down(img_emb[0].unsqueeze(0).unsqueeze(0))

            # emb = self.img_down(img_emb)
            # import ipdb;ipdb.set_trace()
            # emb = self.img_down(img_emb+ cam_emb) 
            # emb = self.downsamnple_conv(emb)
            # emb = self.img_downconv(emb)
           
            
        if self.param_input:
            pose = x[0]
            shape = x[1]
            rot = x[2]
            trans = x[3]
            pose = self.pose(pose)
            shape = self.shape(shape)
            rot = self.rotate(rot)  
            trans = self.trans(trans)
            if len(pose.shape) == 2:
                pose = pose[:,None,:]
                shape = shape[:,None,:]
                rot = rot[:,None,:]
                trans = trans[:,None,:]
            x = torch.cat((pose,shape,rot,trans),dim=1)
            x = self.pre_norm(x)
            if mask is not None:
                x = self.masking(x, mask)
            x = self.pre_conv(x)
            
        else:
            if x.shape[-1] != self.pose_dim:
                mask = x[:, :, -1] == 0
                x = x[:, :, :-1]
            x = self.pre_emb(x)
            x = self.pre_norm(x)
        
            if mask is not None:
                x = self.masking(x, mask)
            # import ipdb;ipdb.set_trace()
            x = self.pre_conv(x)
        if self.dino and not self.cross_attention:
            x = torch.cat((emb, x), dim=1)
            # x = self.pre_conv(x)  # 6890+4096 -> 2000
        # import ipdb;ipdb.set_trace()    
        # import ipdb;ipdb.set_trace()
        x = x+self.positional_emb
        x = self.dropout(x)
        
        
        if self.cross_attention:
            x = self.encoder(x,context=emb)
        else:
            x = self.encoder(x)

        
        if self.multi_view>1:
            
            # import ipdb;ipdb.set_trace()
            
            # torch.nn.functional.avg_pool1d(x, kernel_size=self.multi_view, channels=1)
            import ipdb;ipdb.set_trace()
            x = torch.transpose(x, 0, 1)
            x = self.merge_views(x)
            
            x = torch.transpose(x, 0, 1)
        if self.upsample != 1:
            x = self.upsample_conv(x) # 2000+4096 -> 6890*n

        
        means3D = self.mean3D_head(x)
        opacity = self.opacity_head(x).sigmoid()
        shs = self.shs_head(x)
        rotations = self.rotations_head(x).sigmoid()
        scales = self.scales_head(x).sigmoid()
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
