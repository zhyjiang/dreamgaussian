import os
import cv2
from matplotlib import pyplot as plt
import time
import skimage
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, ViTConfig
from sh_utils import eval_sh, SH2RGB, RGB2SH


import rembg

from lib.gs.gs_renderer import Renderer,Camera
from kornia.losses.ssim import SSIMLoss, ssim_loss

import copy
from grid_put import mipmap_linear_grid_put_2d
from lib.model.UV_GS import UV_GS
from lib.dataset.ZJU import ZJU
from lib.dataset.HuMMan import HuMManDatasetBatch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from .network import get_network, LinLayers
from .utils import get_state_dict
from math import exp
from torch.autograd import Variable
from collections import OrderedDict

torch.manual_seed(0)
np.random.seed(0)
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity




class IoULoss(torch.nn.Module):
    def __init__(self, smooth=0.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou

        return loss
def cosine_similarity(features,nearest_point,vts,part_weight):
    
    k = nearest_point.shape[1]
    k_features = features[vts[nearest_point][:,:,0],vts[nearest_point][:,:,1]] # 7670, 5, d
    cur_features = features[vts[:,0],vts[:,1]].unsqueeze(1) # 7670, 1, d
    # import ipdb;ipdb.set_trace()
   
    # import ipdb;ipdb.set_trace()
    
    l = torch.sum(F.cosine_similarity(cur_features,k_features,dim=2)* torch.tensor(part_weight).to(features.device)) # 7670 5
    return l
def features_diff(features,nearest_point,vts):
    # import ipdb;ipdb.set_trace()
    k = nearest_point.shape[1]
    k_features = features[vts[nearest_point][:,:,0],vts[nearest_point][:,:,1]] # 7670, 5, d
    cur_features = features[vts[:,0],vts[:,1]].unsqueeze(1) # 7670, 1, d
    # import ipdb;ipdb.set_trace()
    cur_features = cur_features.repeat((1,k,1))
    l = torch.sum(torch.norm((cur_features-k_features),dim=2)) # 7670 5
    return l
def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # import ipdb;ipdb.set_trace()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)   
                

class LPIPS(torch.nn.Module):
    r"""Creates a criteion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'vgg'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'vgg', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)

class Trainer:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        
        if self.opt.dataset_type == 'ZJU':
            self.data_config = self.opt.dataset
        else:
            self.data_config = self.opt.dataset2
        self.lpips = LPIPS()

        
        self.W = self.data_config.W
        self.H = self.data_config.H
        self.near = opt.near
        self.far = opt.far
        self.pretrain_path = opt.pretrain_path
        os.makedirs(os.path.join(self.opt.save_name),exist_ok=True)
        self.write = SummaryWriter(os.path.join(self.opt.save_name))
        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image
        self.vae_tune = self.opt.vae_tune

        # models
        self.device = torch.device("cuda")
        self.nearest_point = torch.tensor(np.load('nearest_point.npy')).to(self.device)
        self.dimension = 3
        self.model = UV_GS(self.opt).to(self.device)
        self.nearest_point = torch.tensor(np.load('nearest_point.npy')).to(self.device)

        self.pretrain_vae = self.opt.pretrain_vae

        # path = '/home/zhongyuj/GauHuman/output/zju_mocap_refine/pureCoreView_386/point_cloud/iteration_1200/point_cloud.ply'
        # plydata = PlyData.read(path)

        # # import ipdb;ipdb.set_trace()
        # self.scales = torch.tensor(np.concatenate([plydata.elements[0]['scale_0'],plydata.elements[0]['scale_1'],plydata.elements[0]['scale_2']])).view((6890,3)).to(self.device).unsqueeze(0)
        # self.rotations = torch.tensor(np.concatenate([plydata.elements[0]['rot_0'],plydata.elements[0]['rot_1'],plydata.elements[0]['rot_2'],plydata.elements[0]['rot_3']])).view((6890,4)).to(self.device).unsqueeze(0)
            
        # self.shs = torch.tensor(np.load('/home/zhongyuj/dreamgaussian/fixed_scales.npy')).cuda()
        
        # vts = torch.tensor(self.model.generator.vts).to(self.device).float()
        # self.nearest_point = self.knn(vts[:, None, :], vts, self.opt.k + 1)[0][:, :, 1:].squeeze(1)
        # self.input_img_torch
        # np.save(f'nearest_point_{self.opt.k}.npy',self.nearest_point.cpu().numpy())
        # import ipdb;ipdb.set_trace()
        
        # import ipdb;ipdb.set_trace()

        


            
        
        

            

        if self.pretrain_path is not None :
            # import ipdb;ipdb.set_trace()
            ckpt = torch.load(self.pretrain_path)
            # m = {k: v for k, v in ckpt.items() if 'shape_model' not in k}
            # import ipdb;ipdb.set_trace()
            self.model.load_state_dict(ckpt,strict=False)
            # import ipdb;ipdb.set_trace()
            
            
            print(f'Loaded pretrained model from {self.pretrain_path}')

        if self.pretrain_vae is not None:

            ckpt = torch.load(self.pretrain_vae)

            desired_layer_keys = OrderedDict((key, value) for key, value in ckpt.items() if key.startswith('_vq_vae'))

            self.model.shape_model.load_state_dict(desired_layer_keys,strict=False)
            # for in desired_layer_keys:

            desired_layer_keys_2 = OrderedDict((key, value) for key, value in ckpt.items() if key.startswith('_decoder'))


            self.model.shape_model.load_state_dict(desired_layer_keys_2,strict=False)

            # import ipdb;ipdb.set_trace()


           

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree,opt=self.opt)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.eval_steps = 1

        if self.opt.dataset_type == 'ZJU':
            self.dataset = ZJU(opt,'train')
            self.test_dataset = ZJU(opt,'test')
            self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                      batch_size=opt.batch_size, 
                                                      shuffle=True, 
                                                      num_workers=opt.dataset.num_workers)
        
            self.testloader = torch.utils.data.DataLoader(self.test_dataset, 
                                                      batch_size=opt.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=opt.dataset.num_workers)
            
        elif self.opt.dataset_type == 'HuMMan':
            self.dataset2 = HuMManDatasetBatch(data_root=self.opt.dataset2.root,split='train')
            
            self.test_dataset2 = HuMManDatasetBatch(data_root=self.opt.dataset2.root,split='train')
            self.dataloader = torch.utils.data.DataLoader(self.dataset2, 
                                                        batch_size=opt.batch_size, 
                                                        shuffle=True, 
                                                        num_workers=opt.dataset.num_workers)
            self.testloader = torch.utils.data.DataLoader(self.test_dataset2, 
                                                        batch_size=opt.batch_size, 
                                                        shuffle=False, 
                                                        num_workers=opt.dataset.num_workers)
        
        
        
        print(f'data loader size: {len(self.dataloader.dataset)} and test data loader size: {len(self.testloader.dataset)}')
        
        self.reconstruct_loss = None
        self.reg_loss = None
        self.IOU_loss = None
        self.renderer.gaussians.max_radii2D = torch.zeros((self.opt.batch_size,self.opt.upsample*6890, 3)).to(self.device)
        self.xyz_gradient_accum = torch.zeros((self.opt.batch_size,self.opt.upsample*6890, 1), device="cuda")
        self.denom = torch.zeros((self.opt.batch_size,self.opt.upsample*6890, 1), device="cuda")
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()
            
    def get_image_loss(self,ssim_weight=0.2, type="l1"):
        if type == "l1":
            base_loss_fn = torch.nn.functional.l1_loss
            return base_loss_fn
        elif type == "l2":
            base_loss_fn = torch.nn.functional.mse_loss
            return base_loss_fn
        else:
            raise NotImplementedError
        

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
        
    def psnr_metric(self,img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr
        

    def lr_lambda(self,epoch):
        if epoch < self.opt.warmup:
            return self.opt.base_lr + (self.opt.lr - self.opt.base_lr) / self.opt.warmup * epoch
            
        else:
            return self.opt.lr 
        
    def weight_warmup(self,epoch):
        return self.opt.scale_constrain_weight-(epoch*(7)/200)
        

    

    def prepare_train_opt(self):
    
        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        

        

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
       
    
    def prepare_train(self):

        self.step = 0
        

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        
        # self.model.to(self.device)
        
        self.scale_loss = torch.nn.MSELoss()
        self.rotation_loss = torch.nn.MSELoss()
        
        if self.opt.loss.startswith('ssim'):
            subloss = self.opt.loss.split('_')[-1]
            self.reconstruct_loss = self.get_image_loss(ssim_weight=self.opt.ssim_weight, type=subloss)  
            print(f'Using ssim {subloss} loss.')
        elif self.opt.loss=='l2':
            self.reconstruct_loss = torch.nn.MSELoss()
            print('Using l2 loss.')
        elif self.opt.loss=='l1':
            self.reconstruct_loss = torch.nn.L1Loss()
            print('Using l1 loss.')
        else:
            print('No loss defined.')
            raise NotImplementedError
        self.mask_loss = torch.nn.MSELoss()
        if self.opt.reg_loss == 'l2':
            self.reg_loss = torch.nn.MSELoss()
            print('Using l2 reg loss.')
        elif self.opt.reg_loss == 'l1':
            self.reg_loss = torch.nn.L1Loss()
            print('Using l1 reg loss.')
        else:
            print('No reg loss defined.')
            
            
        self.ssim_loss = ssim
        self.smooth_loss = cosine_similarity
        # self.smooth_loss = features_diff
        # self.part_weight = np.load('part_weight.npy')
        self.feature_loss = torch.nn.MSELoss()
        # self.IOU_loss = IoULoss()
     
     
        
            
    def eval_print(self,ep):
        os.makedirs(self.opt.vis_path,exist_ok=True)
        os.makedirs(self.opt.vis_depth_path,exist_ok=True)
        
        torch.manual_seed(0)
        np.random.seed(0)
        
        
        
        with torch.no_grad():
            self.model.eval()
            # self.opt.train_strategy = 'uv'
            
            self.init = True if self.opt.pretrain_uv else False
        
            
            for epoch in range(self.eval_steps):
                psnr_l = []
                ssim_l = []
                lpips_res= [] 
                
                pbar = tqdm.tqdm(self.testloader)
                for ite, data in enumerate(pbar):
                    vertices = data['vertices'].float().to(self.device)
                    gt_images = data['image'].view((-1,self.H,self.W,3)).float().to(self.device)
                    init_scales = data['scale'].float().to(self.device)
                    
                    
                    
                    vertices = vertices.view((-1,6890,3))
                    data['fovy'] = data['fovy'].view((-1))
                    data['fovx'] = data['fovx'].view((-1))
                    data['K'] = data['K'].view((-1,3,3))
                    data['T'] = data['T'].view((-1,1,3))
                    data['R'] = data['R'].view((-1,3,3))
                    init_scales = init_scales.view((-1,6890,3))

                    
                    

                    if self.opt.param_input:
                        smpl_params = data['smpl_param']
                        pose = smpl_params['poses'].float().to(self.device)
                        shape = smpl_params['shapes'].float().to(self.device)
                    
                    bs = data['image'].shape[0]

                    if self.opt.multi_view == 1 and len(gt_images.shape) == 4:
                        if self.opt.crop_image:
                            cur_corners = torch.clamp(corners[0],0,1024)
                            
                            temp_gt_images = gt_images[0][cur_corners[1][1]:cur_corners[0][1],cur_corners[1][0]:cur_corners[0][0]].unsqueeze(0)
                            # import ipdb;ipdb.set_trace()
                            # cv2.imwrite(os.path.join(self.opt.vis_path,f'{ite}_cropped.jpg'), temp_gt_images[0].cpu().numpy()*255.0)
                        else:
                            temp_gt_images = gt_images[0].unsqueeze(0)
                        
                     # means3D, opacity, scales, shs, rotations,features,features_all = self.model(pose,shape,temp_gt_images,cam=None)
                    if self.opt.train_strategy == 'uv':
                        if self.opt.pretrain_uv:
                            shs = self.model(pose,shape,temp_gt_images,cam=None,pretrain_uv=True)
                        else:
                            means3D, opacity, scales, shs, rotations = self.model(pose,shape,temp_gt_images,cam=None)
                    else:
                        means3D, opacity, scales, shs, rotations = self.model(pose,shape,temp_gt_images,cam=None)
                    if self.opt.train_strategy == 'uv':
                        if self.init:
                            means3D = torch.zeros((bs,6890,3)).to(self.device)
                            # scales =  torch.mean(torch.exp(self.scales),dim=1).repeat(1,6890,1)
                            scales = torch.ones((bs,6890,3)).to(self.device)*0.01
                            opacity = torch.ones((bs,6890,1)).to(self.device)
                            # scales = torch.exp(self.scales) 
                            rotations = torch.zeros((bs,6890,4)).to(self.device)
                            rotations[...,0] = 1 
                            # rotations = torch.mean(F.normalize(self.rotations),dim=1).repeat(1,6890,1)
                        else:
                            scales  = scales * self.opt.scale_factor

                            means3D = means3D * self.opt.means3D_scale
                    
                    else:
                        # opacity = torch._like(opacity)
                        scales  = scales * self.opt.scale_factor
                        means3D = means3D * self.opt.means3D_scale
                    
                    if self.opt.upsample != 1:
                        vertices = vertices.repeat( 1,self.opt.upsample, 1)
                    if self.opt.triple_point:
                        import ipdb;ipdb.set_trace()
                        extra_points = self.model.generator.midpoints_to_v[:,0,:] @ vertices  #13776 , 3
                        vertices = torch.concatenate([vertices,extra_points],dim=0)
                  
                    if self.opt.learned_scale:
                        scales = scales * self.model.model.ln_scale_weight.sigmoid()*self.opt.learned_scale_weight
                    
                    # if ite == 0:
                    #     plt.hist(scales[...,0].flatten().detach().cpu().numpy(),bins=10)
                    #     plt.savefig(f'{self.opt.vis_ply_path}/scale_0.png')
                    #     plt.close()
                    #     plt.hist(scales[...,1].flatten().detach().cpu().numpy(),bins=10)
                    #     plt.savefig(f'{self.opt.vis_ply_path}/scale_1.png')
                    #     plt.close()

                    #     plt.hist(scales[...,2].flatten().detach().cpu().numpy(),bins=10)
                    #     plt.savefig(f'{self.opt.vis_ply_path}/scale_2.png')
                    #     plt.close()
                    

                    for i in self.opt.dataset.test_camera_list:
                        idx = i-1
                        cam = Camera(
                            R = data['R'][idx],
                            T= data['T'][idx],
                            K= data['K'][idx],
                            W=224 if self.opt.superresolution else self.W ,
                            H=224 if self.opt.superresolution  else self.H,
                            FoVy = data['fovy'][idx],
                            FoVx = data['fovx'][idx],
                        )
                        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                        out = self.renderer.render(cam,
                                            self.model.model.w@vertices[idx]+means3D[0] if self.opt.learned_w else vertices[idx]+means3D[0],
                                            opacity[0],
                                            init_scales[idx]+scales[0],
                                            shs[0][:, None, :],
                                            rotations[0],
                                            bg_color=bg_color)
                       
                        image = out["image"] # [1, 3, H, W] in [0, 1]
                        # depth = out['depth'].squeeze() # [H, W]
                        # import ipdb;ipdb.set_trace()
                        if self.opt.superresolution:
                            image = self.model.model.superresolution(image)
                            image = image.squeeze(0)
                        
                        np_img = image.detach().cpu().numpy()
                        np_img = np_img.transpose(1, 2, 0)
                        
                        target_img = gt_images[idx].cpu().numpy()
                        # target_img = target_img.transpose(1, 2, 0)

                        # depth_img = depth.detach().cpu().numpy()
                        #     cv2.imwrite(f'./vis_temp/{ite}.jpg', np.concatenate((target_img, np_img), axis=1) * 255)
                        # plt.imsave(os.path.join(self.opt.vis_eval_depth_path,f'{ite}.jpg'), depth_img)
                        # cv2.imwrite(os.path.join(self.opt.vis_eval,f'{ite}.jpg'), np.concatenate((target_img, np_img), axis=1) * 255)
                        cv2.imwrite(f'{self.opt.vis_novel_view}/{ite}_view{idx}.jpg', np.concatenate((target_img[...,::-1], np_img[...,::-1]), axis=1) * 255)
                        psnr_l.append(self.psnr_metric(np_img, target_img))
                        ssim_l.append(skimage.metrics.structural_similarity( target_img,np_img, multichannel=True,channel_axis=-1,data_range=1.0))  
                        if self.lpips is not None and (i in self.opt.dataset.test_camera_list[:]):
                            self.lpips.eval()
                            lpips_res.append(self.lpips(torch.tensor(np_img).permute(2,0,1), torch.tensor(target_img).permute(2,0,1)).item()) 
                            
                        with torch.no_grad(): 
                            shs_view = shs.detach().view(
                                -1, 3, (0 + 1) ** 2
                            )
                            dir_pp = vertices[idx]+means3D - cam.camera_center.repeat(
                                shs.shape[0], 1
                            )
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            sh2rgb = eval_sh(
                                0, shs_view, dir_pp_normalized
                            )
                            
                            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                            uv_map = torch.zeros_like(gt_images[idx])
                            # import ipdb;ipdb.set_trace()
                            self.model.generator.draw_uv_map(vertices[idx]+means3D[0].detach(),colors_precomp,uv_map,ep,ite,self.opt.vis_ply_path,mode='whole')            
                print('psnr_eval', np.array(psnr_l).mean())  
                print('ssim_eval', np.array(ssim_l).mean()) 
                print('lpips_eval',np.array(lpips_res).mean())    
                self.save_ply(self.opt.vis_ply_path,vertices[idx]+means3D[0],opacity[0],scales[0],rotations[0],shs[0],ite) 
                import ipdb;ipdb.set_trace()
    def eval(self,ep):
        os.makedirs(self.opt.vis_path,exist_ok=True)
        os.makedirs(self.opt.vis_depth_path,exist_ok=True)
        
        
        with torch.no_grad():
            self.model.eval()
        
            
            for epoch in range(self.eval_steps):
                psnr_l = []
                ssim_l = []
                lpips_res = []
                psnr_l_normed = []
                np_imgs = [] 
                gt_imgs = []
                
                pbar = tqdm.tqdm(self.testloader)
                for ite, data in enumerate(pbar):
                    vertices = data['vertices'].float().to(self.device)
                    init_scales = data['scale'].float().to(self.device)
                    init_rotations = data['rotations'].float().to(self.device)
                    gt_images = data['image'].view((-1,self.H,self.W,3)).float().to(self.device)
                    
                    if data['fake'] is not None and len(data['fake']) > 0: 
                    
                
                        fake_images = data['fake'].float().to(self.device).view((-1,self.H,self.W,3))
                    else:
                        fake_images = None
                    vertices = vertices.view((-1,6890,3))
                    init_scales = init_scales.view((-1,6890,3))
                    init_rotations = init_rotations.view((-1,6890,4))
                    data['fovy'] = data['fovy'].view((-1))
                    data['fovx'] = data['fovx'].view((-1))
                    data['K'] = data['K'].view((-1,3,3))
                    data['T'] = data['T'].view((-1,1,3))
                    data['R'] = data['R'].view((-1,3,3))
                    corners = data['corners'].view((-1,2,2))
                    bs = gt_images.shape[0]
                    
                    if self.opt.param_input:
                        smpl_params = data['smpl_param']
                    
                        pose = smpl_params['poses'].float().to(self.device)
                        shape = smpl_params['shapes'].float().to(self.device)
                    
                    if self.opt.multi_view == 1 and len(gt_images.shape) ==4:
                        if self.opt.crop_image:
                            cur_corners = torch.clamp(corners[0],0,1024)
                            temp_gt_images = gt_images[0][cur_corners[1][1]:cur_corners[0][1],cur_corners[1][0]:cur_corners[0][0]].unsqueeze(0)
                        else:
                            
                            temp_gt_images = gt_images[0].unsqueeze(0)
                            
                  
                    # means3D, opacity, scales, shs, rotations,features,features_all = self.model(pose,shape,temp_gt_images,cam=None)
                    if self.opt.train_strategy == 'uv':
                        if self.opt.pretrain_uv:
                            shs = self.model(pose,shape,temp_gt_images,cam=None,pretrain_uv=True)
                        else:
                            means3D, opacity, scales, shs, rotations = self.model(pose,shape,temp_gt_images,cam=None)
                    else:
                        means3D, opacity, scales, shs, rotations = self.model(pose,shape,temp_gt_images,cam=None)
                        

                    if self.opt.train_strategy == 'uv':
                        if self.init:
                            means3D = torch.zeros((bs,6890,3)).to(self.device)
                            scales = torch.ones((bs,6890,3)).to(self.device)*0.01
                            opacity = torch.ones((bs,6890,1)).to(self.device)
                            rotations = torch.zeros((bs,6890,4)).to(self.device)
                            rotations[...,0] = 1 
                        else:
                            scales  = scales * self.opt.scale_factor
                            means3D = means3D * self.opt.means3D_scale
                    
                    
                    else:
                        # scales  = scales * self.opt.scale_factor
                        means3D = means3D * self.opt.means3D_scale

                    scales = torch.exp(scales).clamp(min=0,max=None)
                    rotations = F.normalize(rotations)

                        
                    # opacity = 0.1*torch.ones(opacity)
                    # opacity = 0.1*torch.ones((means3D.shape[0],means3D.shape[1],1))
                    
                        
                    if self.opt.triple_point:
                        extra_points = self.model.generator.midpoints_to_v[:,0,:] @ vertices  #13776 , 3
                        vertices = torch.concatenate([vertices,extra_points],dim=1)
                
                    if self.opt.learned_scale:
                        scales = scales * self.model.ln_scale_weight.sigmoid()*self.opt.learned_scale_weight
                    
                    if ite == 0:
                        try:
                            plt.hist(scales[...,0].flatten().detach().cpu().numpy(),bins=10)
                            plt.savefig(f'{self.opt.vis_ply_path}/scale_0.png')
                            plt.close()

                            plt.hist(scales[...,1].flatten().detach().cpu().numpy(),bins=10)
                            plt.savefig(f'{self.opt.vis_ply_path}/scale_1.png')
                            plt.close()

                            plt.hist(scales[...,2].flatten().detach().cpu().numpy(),bins=10)
                            plt.savefig(f'{self.opt.vis_ply_path}/scale_2.png')
                            plt.close()
                        except:
                            pass

                    for i in self.opt.dataset.test_camera_list:
                        idx = i-1
                        cam = Camera(
                            R = data['R'][idx],
                            T= data['T'][idx],
                            K= data['K'][idx],
                            W=224 if self.opt.superresolution else self.W ,
                            H=224 if self.opt.superresolution  else self.H,
                            FoVy = data['fovy'][idx],
                            FoVx = data['fovx'][idx],
                        )
                        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                        

                        

                        out = self.renderer.render(cam,
                                                self.model.model.w@vertices[idx]+means3D[0] if self.opt.learned_w else vertices[idx]+means3D[0],
                                                opacity[0],
                                                scales[0],
                                                shs[0][:, None, :],
                                                rotations[0],
                                                bg_color=bg_color)
                        image = out["image"] # [1, 3, H, W] in [0, 1]
                        if self.opt.superresolution:
                            image = self.model.model.superresolution(image)
                            image = image.squeeze(0)
                        depth = out['depth'].squeeze() # [H, W]
                        
                        np_img = image.detach().cpu().numpy()
                        np_img = np_img.transpose(1, 2, 0)
                        
                        target_img = gt_images[idx].cpu().numpy()
                        try:
                            plt.imsave(f'{self.opt.vis_novel_view}/{ite}_view{idx}.jpg', np.concatenate((target_img, np_img), axis=1) )
                        except:
                            pass
                        psnr_l.append(self.psnr_metric(np_img, target_img))
                        ssim_l.append(skimage.metrics.structural_similarity( target_img,np_img, multichannel=True,channel_axis=-1,data_range=1.0))   

                        if self.lpips is not None and i in self.opt.dataset.test_camera_list[:3]:
                            self.lpips.eval()
                            lpips_res.append(self.lpips(torch.tensor(np_img).permute(2,0,1), torch.tensor(target_img).permute(2,0,1)).item())
                        
                        # with torch.no_grad(): 
                        #     shs_view = shs.detach().view(
                        #         -1, 3, (0 + 1) ** 2
                        #     )
                        #     dir_pp = vertices[idx]+means3D - cam.camera_center.repeat(
                        #         shs.shape[0], 1
                        #     )
                        #     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        #     sh2rgb = eval_sh(
                        #         0, shs_view, dir_pp_normalized
                        #     )
                            
                        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                        #     uv_map = torch.zeros_like(gt_images[idx])
                            
                        #     import ipdb;ipdb.set_trace()
                        #     self.model.generator.draw_uv_map(vertices[idx]+means3D[0].detach(),colors_precomp,uv_map,ep,ite,self.opt.vis_ply_path,mode='pixel_only')            
                        #     self.model.generator.draw_uv_map(vertices[idx]+means3D[0].detach(),colors_precomp,uv_map,ep,ite,self.opt.vis_ply_path,mode='whole')            
                self.write.add_scalar('psnr_eval', np.array(psnr_l).mean(),global_step=ep)        
                self.write.add_scalar('ssim_eval', np.array(ssim_l).mean(),global_step=ep)
                if self.lpips is not None:
                    self.write.add_scalar('lpips_eval',np.array(lpips_res).mean(),global_step=ep)

    
    
    def train_step(self,ep):
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        self.init = True if self.opt.pretrain_uv else False
        
        linear_schedule = torch.arange(0.1,0,-0.01)
        if not self.opt.train_strategy == 'both':
            if ep % self.opt.switch_epoch == 0:
                if self.opt.train_strategy == 'uv' :
                    self.opt.train_strategy = 'shape'
                    self.model.freeze(self.model.uv_model)
                    self.model.activate(self.model.shape_model)

                else:
                    self.opt.train_strategy = 'uv'
                    self.model.activate(self.model.uv_model)
                    if not self.opt.pretrain_uv:
                        self.model.freeze(self.model.shape_model)
        else:
            self.model.activate(self.model.uv_model)
            self.model.activate(self.model.shape_model)

            if self.vae_tune:
                self.model.activate(self.model.shape_model._decoder)
                self.model.activate(self.model.shape_model._vq_vae)
            else:
                self.model.freeze(self.model.shape_model._decoder)
                self.model.freeze(self.model.shape_model._vq_vae)
                self.model.check_no_grad(self.model.shape_model._decoder)
                self.model.check_no_grad(self.model.shape_model._vq_vae)

            
        world_vertex = {}
        for epoch in range(self.train_steps):
            
            os.makedirs(self.opt.vis_path,exist_ok=True)
            os.makedirs(self.opt.vis_depth_path,exist_ok=True)
            os.makedirs(self.opt.vis_novel_view,exist_ok=True)
            print(self.optimizer.param_groups[0]['lr'])
            self.model.train()

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)
            overall_loss= 0
            overall_image_loss =0 
            overall_mask_loss =0 
            overall_ssim_loss =0 
            
          
            # update lr
            pbar = tqdm.tqdm(self.dataloader)
            for ite, data in enumerate(pbar):

                loss = 0
                self.optimizer.zero_grad()
                
                image_name = data['image_path']
                
                vertices = data['vertices'].float().to(self.device)
                init_scales = data['scale'].float().to(self.device)
                init_rotations = data['rotations'].float().to(self.device)
                
                if 'smpl_param' in data.keys():
                
                    smpl_params = data['smpl_param']
                    pose = smpl_params['poses'].float().to(self.device)
                    shape = smpl_params['shapes'].float().to(self.device)

                gt_images = data['image'].float().to(self.device).view((-1,self.H,self.W,3))
                
                if data['fake'] is not None and len(data['fake']) > 0: 
                    
                
                    fake_images = data['fake'].float().to(self.device).view((-1,self.H,self.W,3))
                else:
                    fake_images = None
                
                mask = data['mask'].float().to(self.device).view((-1,self.H,self.W))
                bound_mask = data['bound_mask'].float().to(self.device).view((-1,self.H,self.W))
                vertices = vertices.view((-1,6890,3))
                init_scales = init_scales.view((-1,6890,3))
                init_rotations = init_rotations.view((-1,6890,4))
                corners = data['corners'].view((-1,2,2))
               
                data['K'] = data['K'].view((-1,3,3))
                data['T'] = data['T'].view((-1,1,3))
                
                data['R'] = data['R'].view((-1,3,3))
                
                data['fovy'] = data['fovy'].view((-1))
                data['fovx'] = data['fovx'].view((-1))
                cam_list= [ ]
                
                bs = data['image'].shape[0]
                
                train_camera_set = copy.copy(self.opt.dataset.train_camera_list)
                

                rand_view = 0
                if self.opt.multi_view == 1 and len(gt_images.shape) == 4:
                    if self.opt.crop_image:
                        cur_corners = torch.clamp(corners[0],0,1024)
                        temp_gt_images = gt_images[rand_view][cur_corners[1][1]:cur_corners[0][1],cur_corners[1][0]:cur_corners[0][0]].unsqueeze(0)
                    else:
                        if fake_images.shape[0] != 0  and self.opt.smpl_driven_training:
                            temp_gt_images = fake_images[rand_view].unsqueeze(0)
                        else:
                            temp_gt_images = gt_images[rand_view].unsqueeze(0)
                            
                if self.opt.train_strategy == 'uv':
                    # self.model.freeze(self.model.shape_model)
                    # self.model.activate(self.model.uv_model)
                    if self.opt.pretrain_uv:
                        shs = self.model(pose,shape,temp_gt_images,cam=None,pretrain_uv=True)
                    else:
                        means3D, opacity, scales, shs, rotations = self.model(pose,shape,temp_gt_images,cam=None)

                else:
                    means3D,opacity, scales, shs, rotations = self.model(pose,shape,temp_gt_images,cam=None)
                    
                # import ipdb;ipdb.set_trace()
                # shs = shs.clamp(0,1)
 
                if self.opt.triple_point:
                    extra_points = self.model.generator.midpoints_to_v[:,0,:] @ vertices  #13776 , 3
                    vertices = torch.concatenate([vertices,extra_points],dim=1)
                
                
                if self.opt.train_strategy == 'uv':
                    if self.init:
                        means3D = torch.zeros((bs,6890,3)).to(self.device)
                        scales = torch.ones((bs,6890,3)).to(self.device)*0.00
                        opacity = torch.ones((bs,6890,1)).to(self.device)
                        rotations = torch.zeros((bs,6890,4)).to(self.device)
                        # rotations[...,0] = 1 
                    else:
                        scales  = scales * self.opt.scale_factor

                        means3D = means3D * self.opt.means3D_scale
                    
                    
                else:
                    # scales  = scales * self.opt.scale_factor
                    means3D = means3D * self.opt.means3D_scale
                    
                scales = torch.exp(scales).clamp(min=0,max=None)
                rotations = F.normalize(rotations)
                # if self.opt.learned_scale:
                #     scales = scales * self.model.ln_scale_weight.sigmoid()*self.opt.learned_scale_weight
             
                for i in train_camera_set:  

                    idx = i-1
                    cam = Camera(
                        R = data['R'][idx],
                        T= data['T'][idx],
                        K= data['K'][idx],
                        W=224 if self.opt.superresolution else self.W ,
                        H=224 if self.opt.superresolution  else self.H,
                        FoVy = data['fovy'][idx],
                        FoVx = data['fovx'][idx],
                    )

                    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                    # import ipdb;ipdb.set_trace()
                    # if scales.min() < 0:
                    #     import ipdb;ipdb.set_trace()

                    try:
                        out = self.renderer.render(cam,
                                                self.model.model.w@vertices[idx]+means3D[0] if self.opt.learned_w else vertices[idx]+means3D[0],
                                                opacity[0],
                                                scales[0],
                                                # scales[0],
                                                shs[0][:, None, :],
                                                rotations[0],
                                                bg_color=bg_color)
                    except:
                        import ipdb;ipdb.set_trace()
                        
                   
                    image = out["image"] # [1, 3, H, W] in [0, 1]
                    if self.opt.superresolution:
                        image = self.model.model.superresolution(image)
                        image = image.squeeze(0)

                    msk = out['alpha']
                    
                    if self.opt.train_strategy != 'uv':
                        loss = loss + self.reg_loss(means3D[0], torch.zeros_like(means3D[0])) * self.opt.smpl_reg_scale
                    mask_loss = self.mask_loss(msk[0][bound_mask[idx]==1],mask[idx][bound_mask[idx]==1]) * self.opt.mask_weight
                    loss = loss + mask_loss
                    
                    # if self.opt.scale_loss and self.opt.scale_constrain_weight != 0:
                    #     loss = loss + self.scale_loss(torch.std(scales,dim=-1), torch.zeros_like(torch.std(scales,dim=-1))) * self.opt.scale_constrain_weight
                        # import ipdb;ipdb.set_trace()
                        # loss = loss + self.scale_loss(torch.std(scales,dim=1), torch.zeros_like(torch.std(scales,dim=1))) * 20.0
                    overall_mask_loss += mask_loss.item()
                    
                    image_loss = self.reconstruct_loss(image.permute((1,2,0))[bound_mask[idx]==1,:], gt_images[idx][bound_mask[idx]==1,:])
                    # loss = loss + (5.0-(self.smooth_loss(features,self.nearest_point,self.model.generator.vts,self.part_weight) / self.nearest_point.shape[0]))* self.opt.similarity_score_weight
                    loss = loss + image_loss
                    overall_image_loss += image_loss.item()
                    
                    x, y, w, h = cv2.boundingRect(bound_mask[idx].cpu().numpy().astype(np.uint8))
                    img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
                    img_gt = gt_images[idx][ y:y + h, x:x + w,:].unsqueeze(0).permute((0,3,1,2))
                    if not self.opt.dino_interpolation:
                        ssim_loss =  (1.0 - self.ssim_loss(img_pred,img_gt))*self.opt.ssim_weight
                        overall_ssim_loss += ssim_loss.item()
                        loss = loss + ssim_loss
                    
                   
                    
                    if ite % 30 == 0 or ite == 1:
                        np_img = image.detach().cpu().numpy().transpose(1, 2, 0)
                        target_img = gt_images[idx].cpu().numpy()
                        try:
                            plt.imsave(os.path.join(self.opt.vis_path,f'{ite}_view{idx}.jpg'), np.concatenate((target_img, np_img), axis=1))
                        except:
                            pass

                    if ep %4 == 0 and ite%20 == 0:
                        try:
                            with torch.no_grad():
                                self.save_ply(self.opt.vis_ply_path,vertices[idx]+means3D[0],opacity[0],init_scales[0]+scales[0],rotations[0],shs[0],ite,ep=ep)
                        except:
                            pass 
                        
                        if i == self.opt.dataset.train_camera_list[0]:
                            shs_view = shs.detach().view(
                                -1, 3, (0 + 1) ** 2
                            )
                            dir_pp = means3D - cam.camera_center.repeat(
                                shs.shape[0], 1
                            )
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            sh2rgb = eval_sh(
                                0, shs_view, dir_pp_normalized
                            )
                            
                            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                            uv_map = torch.zeros_like(gt_images[idx])
                            self.model.generator.draw_uv_map(vertices[idx]+means3D[0].detach(),colors_precomp,uv_map,ep,ite,self.opt.vis_ply_path,mode='point_only')   
                            self.model.generator.draw_uv_map(vertices[idx]+means3D[0].detach(),colors_precomp,uv_map,ep,ite,self.opt.vis_ply_path,mode='p')             
                        
                        
                          
                                
          
                pbar.set_postfix({'Loss': f'{loss.item():.5f}', 
                                  'DB': f'{scales.mean().item():.5f}','var': f'{scales.var().item():.5f}'})
              
                loss.backward()
                overall_loss+=loss.item()
                self.optimizer.step()
                self.write.add_scalar('Train/Loss', loss.item()/len(data['R']), global_step=ep*len(self.dataloader)+ite)  
                
            self.opt.smpl_reg_scale *= 0.7
            self.scheduler.step()
            self.write.add_scalar('Train/Loss Epoch', overall_loss/(len(self.dataloader)*len(data['R'])), global_step=ep) 
            self.write.add_scalar('SSIM Train/Loss Epoch', overall_ssim_loss/(len(self.dataloader)*len(data['R'])), global_step=ep) 
            self.write.add_scalar('Image Train/Loss Epoch', overall_image_loss/(len(self.dataloader)*len(data['R'])), global_step=ep) 
            self.write.add_scalar('MASK Train/Loss Epoch', overall_mask_loss/(len(self.dataloader)*len(data['R'])), global_step=ep) 
            
                

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )
            
        

   
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l
    
    def inverse_sigmoid(self,x):
        return np.log(x/(1-x))
        
    def save_ply(self, path, xyz, opacity, scale, rotation, color, idx,ep=None):
        os.makedirs(path,exist_ok=True)
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_rest = torch.zeros((xyz.shape[0],45)).cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

        dtype_full = [(attribute,'f4') for attribute in self.construct_list_of_attributes()]
        

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # import ipdb;ipdb.set_trace()
        # if self.opt.exp_scale:
        attributes = np.concatenate((xyz, normals, color, f_rest,self.inverse_sigmoid(opacities), torch.log(scale),rotation), axis=1)
        # else:
        #     # import ipdb;ipdb.set_trace()
        #     attributes = np.concatenate((xyz, normals, color, f_rest,self.inverse_sigmoid(opacities), self.inverse_sigmoid(scale),rotation), axis=1)
            
        

   
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        # print(f'ply saved in {path}')
        if ep is not None:
            PlyData([el]).write(os.path.join(path,f'epoch_{ep}_{idx}.ply'))
        else:
            
            PlyData([el]).write(os.path.join(path,f'{idx}.ply'))

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            # mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            # mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iteations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iteations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            # self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
                
            self.test_step()
            
            
            dpg.render_dearpygui_frame()
            
        
    
    # no gui mode
    def train(self, ites=500):
        if ites > 0:
            self.prepare_train()
            # prepare_train_3DGS()
            
            # if self.post_3DGS:
               
            #     self.prepare_train_opt()
            with torch.no_grad():
                if self.opt.eval_only:
                    self.eval_print(0)
                    import ipdb;ipdb.set_trace()
            
                
                
            for i in tqdm.trange(ites):
                values = self.train_step(i)
                # if self.opt.post_3DGS:
                #     train_step_3DGS(i,values)

                if i % self.opt.eval_interval == 0:
                    os.makedirs(os.path.join(self.opt.vis_eval_depth_path),exist_ok=True)
                    os.makedirs(os.path.join(self.opt.vis_eval),exist_ok=True)
                    self.eval(i)
               
                if i % self.opt.save_interval == 0:
                    # self.save_model(mode='model')
                    # os.makedirs(f'./checkpoints/{self.opt.save_path}/',exist_ok=True)
                    # os.makedirs(f'./checkpoints/{self.opt.save_path}/{self.opt.save_name}',exist_ok=True)
                    os.makedirs(f'{self.opt.save_path}/',exist_ok=True)
                    os.makedirs(f'{self.opt.save_path}/{self.opt.save_name}',exist_ok=True)
                    try:
                        with torch.no_grad():
                            # torch.save(self.model.state_dict(), f'./checkpoints/{self.opt.save_path}/{self.opt.save_name}/epoch{i}.pth')
                            torch.save(self.model.state_dict(), f'{self.opt.save_path}/{self.opt.save_name}/epoch{i}.pth')
                            
                    except:
                        pass
        self.write.close()
            # do a last prune
            # self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save


    def knn(self,query_points, data_points, k=1):

        # Compute pairwise distances between query points and data points
        distances = torch.cdist(query_points, data_points)
        
        # Find the k nearest neighbors
        distances, indices = torch.topk(distances, k, largest=False)
        
        return indices, distances
    