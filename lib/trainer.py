import os
import cv2
from matplotlib import pyplot as plt
import time
import skimage
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from lib.gs.gs_renderer import Renderer, MiniCam
from kornia.losses.ssim import SSIMLoss, ssim_loss

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from lib.model.vertex_encoder import VertexTransformer
from lib.dataset.ZJU import ZJU
from lib.dataset.HuMMan import HuMManDatasetBatch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from .network import get_network, LinLayers
from .utils import get_state_dict
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

class LPIPS(torch.nn.Module):
    r"""Creates a criterion that measures
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
            
        # self.lpips = LPIPS()
        self.lpips = None
        self.W = self.data_config.W
        self.H = self.data_config.H
        self.near = opt.near
        self.far = opt.far
        # self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        os.makedirs(os.path.join('logs',self.opt.save_name),exist_ok=True)
        self.writer = SummaryWriter(os.path.join('logs',self.opt.save_name))
        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        # self.bg_remover = None

        # self.guidance_sd = None
        # self.guidance_zero123 = None

        # self.enable_sd = False
        # self.enable_zero123 = False
        img_emb_dim = self.data_config.img_emb_dim
        
        self.encoder = VertexTransformer(opt=self.opt,upsample=self.opt.upsample,dino=self.opt.dino,img_dim=img_emb_dim,param_input=self.opt.param_input,cross_attention=self.opt.cross_attn,pose_num=self.data_config.pose_num,multi_view=self.opt.multi_view,camera_param=self.opt.camera_param,dino_update = self.opt.dino_update,device=self.device).to(self.device)

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5
        
        
          
            
        # self.alpha = torch.nn.Parameter(torch.tensor(0.5),requires_grad=True).to(self.device)

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        
        

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.eval_steps = 1
        # import ipdb;ipdb.set_trace()
        # self.dataset = ZJU(opt)
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
            # import ipdb;ipdb.set_trace()
            
        
        
            self.dataset2 = HuMManDatasetBatch(data_root=self.opt.dataset2.root,split='train')
            
            self.test_dataset2 = HuMManDatasetBatch(data_root=self.opt.dataset2.root,split='test')
            # self.test_dataset = ZJU(opt,'test')
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
    
    # def IOU_silou_loss(self,mask1,mask2):
    #     # Load mask images as NumPy arrays

    #     # Calculate intersection and union
    #     intersection = np.logical_and(mask1, mask2)
    #     union = np.logical_or(mask1, mask2)

    #     # Compute IoU
    #     iou = np.sum(intersection) / np.sum(union)
    #     return iou
        

    def lr_lambda(self,epoch):
        if epoch < self.opt.warmup:
            return self.opt.base_lr + (self.opt.lr - self.opt.base_lr) / self.opt.warmup * epoch
            
        else:
            return self.opt.lr 
    def prepare_train(self):

        self.step = 0

        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1.0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        
        self.encoder.to(self.device)
        
        
        
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
        
        if self.opt.reg_loss == 'l2':
            self.reg_loss = torch.nn.MSELoss()
            print('Using l2 reg loss.')
        elif self.opt.reg_loss == 'l1':
            self.reg_loss = torch.nn.L1Loss()
            print('Using l1 reg loss.')
        else:
            print('No reg loss defined.')
            
        self.IOU_loss = IoULoss()
        # self.IOU_loss = torch.nn.MSELoss()
        # self.IOU_loss = torch.nn.BCELoss() + DICELoss()
        
            

        # setup training
        # self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        # self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        # self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        # pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        # self.fixed_cam = MiniCam(
        #     pose,
        #     self.opt.ref_size,
        #     self.opt.ref_size,
        #     self.cam.fovy,
        #     self.cam.fovx,
        #     self.cam.near,
        #     self.cam.far,
        # )

        # self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        # self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        # if self.guidance_sd is None and self.enable_sd:
        #     if self.opt.mvdream:
        #         print(f"[INFO] loading MVDream...")
        #         from guidance.mvdream_utils import MVDream
        #         self.guidance_sd = MVDream(self.device)
        #         print(f"[INFO] loaded MVDream!")
        #     else:
        #         print(f"[INFO] loading SD...")
        #         from guidance.sd_utils import StableDiffusion
        #         self.guidance_sd = StableDiffusion(self.device)
        #         print(f"[INFO] loaded SD!")

        # if self.guidance_zero123 is None and self.enable_zero123:
        #     print(f"[INFO] loading zero123...")
        #     from guidance.zero123_utils import Zero123
        #     self.guidance_zero123 = Zero123(self.device)
        #     print(f"[INFO] loaded zero123!")

        # input image
        # if self.input_img is not None:
        #     self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        #     self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        #     self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        #     self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        # with torch.no_grad():

        #     if self.enable_sd:
        #         self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

        #     if self.enable_zero123:
        #         self.guidance_zero123.get_img_embeds(self.input_img_torch)
    
    def eval(self,ep):
        os.makedirs(self.opt.vis_path,exist_ok=True)
        os.makedirs(self.opt.vis_depth_path,exist_ok=True)
        
        self.encoder.eval()
        
        with torch.no_grad():
        
            
            for epoch in range(self.eval_steps):

                

                psnr_l = []
                ssim_l = []
                psnr_l_normed = []
                np_imgs = [] 
                gt_imgs = []
                
                pbar = tqdm.tqdm(self.testloader)
                for iter, data in enumerate(pbar):
                    
                    vertices = data['vertices'].float().to(self.device)
                    
                    gt_images = data['image'].view((-1,3,self.H,self.W)).float().to(self.device)
                    mask = data['mask'].view((-1,self.H,self.W)).float().to(self.device)
                    gt_images = gt_images * mask[:, None, :, :]
                    
                    
                    vertices = vertices.view((-1,6890,3))
                    data['w2c'] = data['w2c'].view((-1,4,4))
                    data['fovy'] = data['fovy'].view((-1))
                    data['fovx'] = data['fovx'].view((-1))
                    
                    cam_list = []
                    full_proj_cam_list = []
                    
                    for i in range(len(data['w2c'])):
                
                            cam = MiniCam(
                                data['w2c'][i],
                                self.W,
                                self.H,
                                data['fovy'][i],
                                data['fovx'][i],
                                self.near,
                                self.far,
                            ) 
                        
                       
                            cam_list.append(cam.projection_matrix.to(self.device))
                            full_proj_cam_list.append(cam.full_proj_transform.to(self.device))


                        
                    cam_list = torch.stack(cam_list,dim=0)
                    full_proj_cam_list = torch.stack(full_proj_cam_list,dim=0)
                    if self.opt.param_input:
                        smpl_params = data['smpl_param']
                    
                        pose = smpl_params['poses'].float().to(self.device)
                        shape = smpl_params['shapes'].float().to(self.device)
                        trans = smpl_params['Th'].float().to(self.device)
                        rot = smpl_params['Rh'].float().to(self.device)   
                    
                    if self.opt.param_input:
                        means3D, opacity, scales, shs, rotations = self.encoder((pose,shape,rot,trans),img=gt_images,cam = full_proj_cam_list)
                    
                    else:
                        means3D, opacity, scales, shs, rotations = self.encoder(vertices,img=gt_images,cam = cam_list)
                    if self.encoder.upsample != 1:
                        vertices = vertices.repeat( 1,self.encoder.upsample, 1)
                    scales = scales * self.opt.scale_factor
                    
                    
                    
                    for idx in range(len(vertices)):
                        
                        
                        
                        cam = MiniCam(
                            data['w2c'][idx],
                            self.W,
                            self.H,
                            data['fovy'][idx],
                            data['fovx'][idx],
                            self.near,
                            self.far,
                        )
                        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                        
                        if self.opt.multi_view > 1:
                            
                            out = self.renderer.render(cam,
                                                    vertices[idx]+means3D[0],
                                                    opacity[0],
                                                    scales[0],
                                                    shs[0][:, None, :],
                                                    rotations[0],
                                                    bg_color=bg_color)
                        else:
                        
                            out = self.renderer.render(cam,
                                                    vertices[idx]+means3D[idx],
                                                    opacity[idx],
                                                    scales[idx],
                                                    shs[idx][:, None, :],
                                                    rotations[idx],
                                                    bg_color=bg_color)
                        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        depth = out['depth'].squeeze() # [H, W]
                        
                        
                        np_img = image[0].detach().cpu().numpy()
                        np_imgs.append(np_img)
                        
                        np_img = np_img.transpose(1, 2, 0)
                        
                        
                        target_img = gt_images[idx].cpu().numpy()
                        
                        gt_imgs.append(target_img)
                        target_img = target_img.transpose(1, 2, 0)
                        
                        
                        
                      
                        #     # masked = target_img * mask[0].cpu().numpy()[...,None]
                            
                        depth_img = depth.detach().cpu().numpy()
                        #     cv2.imwrite(f'./vis_temp/{iter}.jpg', np.concatenate((target_img, np_img), axis=1) * 255)
                        plt.imsave(os.path.join(self.opt.vis_eval_depth_path,f'{iter}.jpg'), depth_img)
                        cv2.imwrite(os.path.join(self.opt.vis_eval,f'{iter}.jpg'), np.concatenate((target_img, np_img), axis=1) * 255)
                        
                        pred_img_norm = np_img / np_img.max()
                        gt_img_norm = target_img / target_img.max()
                        
                        psnr_l.append(self.psnr_metric(np_img, target_img))
                        
                        psnr_l_normed.append(self.psnr_metric(pred_img_norm, gt_img_norm))
  
                        ssim_l.append(skimage.metrics.structural_similarity( gt_img_norm,pred_img_norm, multichannel=True,channel_axis=-1,data_range=1.0))   
                
                
                
                if self.lpips is not None:
                        
                    self.lpips.eval()
    
                            
                        
                            
                    lpips_res = self.lpips(torch.tensor(np.array(np_imgs)), torch.tensor(np.array(gt_imgs)))
                self.writer.add_scalar('psnr_eval', np.array(psnr_l).mean(),global_step=ep)        
                self.writer.add_scalar('psnr_normed_eval', np.array(psnr_l_normed).mean(),global_step=ep)
                self.writer.add_scalar('ssim_eval', np.array(ssim_l).mean(),global_step=ep)
                if self.lpips is not None:
                    self.writer.add_scalar('lpips_eval',lpips_res.item(),global_step=ep)
                
          

                

        

    def train_step(self,ep):
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        linear_schedule = torch.arange(2,0,-0.2)
       
        
        world_vertex = {}
        for epoch in range(self.train_steps):
            
            os.makedirs(self.opt.vis_path,exist_ok=True)
            os.makedirs(self.opt.vis_depth_path,exist_ok=True)
            os.makedirs(self.opt.vis_second_view,exist_ok=True)
            print(self.optimizer.param_groups[0]['lr'])
            self.encoder.train()

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)
            overall_loss= 0
            # self.optimizer.zero_grad()

            # update lr
            # self.renderer.gaussians.update_learning_rate(self.step)
            pbar = tqdm.tqdm(self.dataloader)
            for iter, data in enumerate(pbar):
                
                # import ipdb;ipdb.set_trace()
                loss = 0
                self.optimizer.zero_grad()
                
                
                vertices = data['vertices'].float().to(self.device)
                
                if 'smpl_param' in data.keys():
                
                    smpl_params = data['smpl_param']
                    pose = smpl_params['poses'].float().to(self.device)
                    shape = smpl_params['shapes'].float().to(self.device)
                    trans = smpl_params['Th'].float().to(self.device)
                    rot = smpl_params['Rh'].float().to(self.device)  
                
              
                # bbox = data['bbox']
                
                
                
                
                gt_images = data['image'].float().to(self.device).view((-1,3,self.H,self.W))
                
                
                mask = data['mask'].float().to(self.device).view((-1,self.H,self.W))
                if len(mask.shape) != len(gt_images.shape):
                    gt_images = gt_images * mask[:, None, :, :]
                else:
                    gt_images = gt_images * mask
                vertices = vertices.view((-1,6890,3))
                data['w2c'] = data['w2c'].view((-1,4,4))
                data['fovy'] = data['fovy'].view((-1))
                data['fovx'] = data['fovx'].view((-1))
                
                # import ipdb;ipdb.set_trace()

                cam_list= [ ]
                full_proj_cam_list = [ ]
                world_cam_list = []

                # for i in range(len(data['w2c'])):  
                for i in range(1):
                   
                    cam = MiniCam(
                        data['w2c'][i],
                        self.W,
                        self.H,
                        data['fovy'][i],
                        data['fovx'][i],
                        self.near,
                        self.far,
                    ) 
                    
                    
                  
                    cam_list.append(cam.projection_matrix.to(self.device))
                    full_proj_cam_list.append(cam.full_proj_transform.to(self.device))
                    world_cam_list.append(cam.world_view_transform.to(self.device))
                    
                # crop_imgs = []    
                # temp = torch.zeros((len(bbox),3,1000,1000)).to(self.device)
                # if self.opt.crop_image and bbox is not None:
                    
                #     for index in range(len(bbox)):
                #         import ipdb;ipdb.set_trace()
                #         cropped = gt_images[index,...,bbox[index,2]:bbox[index,0],bbox[index,3]:bbox[index,1]]
                        
                #         x = (900//2 - (bbox[index,0]-bbox[index,2])//2).int()
                #         y = (900//2 - (bbox[index,1]-bbox[index,2])//2).int()
                #         try:
                #             temp[index][...,x:x+(bbox[index,0]-bbox[index,2]), y: y+(bbox[index,1]-bbox[index,3])] = cropped
                #         except:
                #             import ipdb;ipdb.set_trace()
                
                    # crop_imgs = torch.stack(crop_imgs,dim=0)
                cam_list = torch.stack(cam_list,dim=0)
                full_proj_cam_list = torch.stack(full_proj_cam_list,dim=0) 
                world_cam_list = torch.stack(world_cam_list,dim=0)   
                # import ipdb;ipdb.set_trace()
                # gt_images = gt_images.view((-1,3,self.H,self.W))
                if self.opt.param_input:
                    means3D, opacity, scales, shs, rotations = self.encoder((pose,shape,rot,trans),img=gt_images,cam = full_proj_cam_list)
                else:  
                    means3D, opacity, scales, shs, rotations = self.encoder(vertices,img=gt_images,cam = cam_list)
                if self.encoder.upsample != 1:
                    vertices = vertices.repeat( 1,self.encoder.upsample, 1)
                scales = scales * self.opt.scale_factor
                # for idx in range(len(data['w2c'])):
                for idx in range(1):   
                    cam = MiniCam(
                        data['w2c'][idx],
                        self.W,
                        self.H,
                        data['fovy'][idx],
                        data['fovx'][idx],
                        self.near,
                        self.far,
                    )
                
                    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                    
                    
                    if self.opt.multi_view > 1:
                        
                        out = self.renderer.render(cam,
                                                vertices[idx]+means3D[0],
                                                opacity[0],
                                                scales[0],
                                                shs[0][:, None, :],
                                                rotations[0],
                                                bg_color=bg_color)
                        
                    else:    
                        out = self.renderer.render(cam,
                                                vertices[idx]+means3D[idx],
                                                opacity[idx],
                                                scales[idx],
                                                shs[idx][:, None, :],
                                                rotations[idx],
                                                bg_color=bg_color)
                   
                    image = out["image"] # [1, 3, H, W] in [0, 1]
                    depth = out['depth'].squeeze() # [H, W]
                    
                    
                    image_mask = image.clone()
                    image_mask[image_mask>0] = 1
                    
                    if self.reg_loss is not None and epoch == 0:
                        
                        # loss = loss+ self.IOU_loss(image_mask[0],mask[idx]) * 1000
                    #     # loss = loss+ self.reg_loss(means3D[idx], torch.rand_like(means3D[idx])*torch.mean(std)*0.5) *self.opt.smpl_reg_scale 
                        if self.opt.multi_view <=1:
                            
                            loss = loss+ self.reg_loss(means3D[idx], torch.zeros_like(means3D[idx])) *self.opt.smpl_reg_scale
                        else:
                            # import ipdb;ipdb.set_trace()
                            
                            loss = loss+ self.reg_loss(means3D[0], torch.zeros_like(means3D[0])) *self.opt.smpl_reg_scale 
                    else:
                    # import ipdb;ipdb.set_trace()
                        assert len(image_mask[0])==2
                        loss = loss+ self.IOU_loss(image_mask[0],mask[idx]) * self.opt.IOU_weight
                        
                        
                    
                    # import ipdb;ipdb.set_trace()
                    
                    # else:
                        # loss = loss+ self.reg_loss(means3D[idx], torch.zeros_like(means3D[idx])) * 0.1
                        # linear_schedule[min(epoch,19)]
                        
                    # elif self.reg_loss is not None and epoch >2 and epoch <= 5 :
                        # loss = loss+ self.reg_loss(means3D[idx], torch.zeros_like(means3D[idx])) * 0.8
                    # else:
                    #     loss = loss+ self.reg_loss(means3D[idx], torch.zeros_like(means3D[idx])) * 0.2
                    # import ipdb;ipdb.set_trace()
                    if self.reconstruct_loss is not None:
                        loss = loss+self.reconstruct_loss(image * mask[idx:idx+1, None,:, :], gt_images[idx:idx+1])
                    
                    if iter % 100 == 0 and idx == 0:
                        np_img = image.detach().cpu().numpy().transpose(1, 2, 0)
                        # smpl_image = smpl_image[0].detach().cpu().numpy().transpose(1, 2, 0)
                        target_img = gt_images[idx].cpu().numpy().transpose(1, 2, 0)
                        
                        # masked = target_img * mask[0].cpu().numpy()[...,None]
                        
                        depth_img = depth.detach().cpu().numpy()
                        
                        # proj = (cam_list[idx] @ vertices[idx].T).T
                        # proj[:, :2] /= proj[:, 2:]
                        # plt.imshow(target_img)
                        # plt.scatter(data['vertices'].float()[...,0],data['vertices'].float()[...,1],c='r')
                        # import ipdb;ipdb.set_trace()
                        
                        cv2.imwrite(os.path.join(self.opt.vis_path,f'{iter}.jpg'), np.concatenate((target_img, np_img), axis=1)*255.0)
                        plt.imsave(os.path.join(self.opt.vis_depth_path,f'{iter}.jpg'), depth_img)
                        
                        
                        
                    # if iter % 400 ==0:
                        
                    #     cam = MiniCam(
                    #         data['w2c'][-1],
                    #         self.W,
                    #         self.H,
                    #         data['fovy'][-1],
                    #         data['fovx'][-1],
                    #         self.near,
                    #         self.far,
                    #     )
                    #     # import ipdb;ipdb.set_trace()
                            
                            
                            
                    #     bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                    
                    
                    #     if self.opt.multi_view > 1:
                            
                    #         out = self.renderer.render(cam,
                    #                                 vertices[idx]+means3D[0],
                    #                                 opacity[0],
                    #                                 scales[0],
                    #                                 shs[0][:, None, :],
                    #                                 rotations[0],
                    #                                 bg_color=bg_color)
                        
                    #     else:    
                    #         out = self.renderer.render(cam,
                    #                                 vertices[idx]+means3D[idx],
                    #                                 opacity[idx],
                    #                                 scales[idx],
                    #                                 shs[idx][:, None, :],
                    #                                 rotations[idx],
                    #                                 bg_color=bg_color)
                    
                    #     image = out["image"] # [1, 3, H, W] in [0, 1]
                    #     depth = out['depth'].squeeze() # [H, W]
                        
                        
                    #     image_mask = image.clone()
                    #     image_mask[image_mask>0] = 1
                        
                        
                    #     np_img = image.detach().cpu().numpy().transpose(1, 2, 0)
                    #     target = gt_images[-1].cpu().numpy().transpose(1, 2, 0)
                    #     # target_img = self_sec[idx].cpu().numpy().transpose(1, 2, 0)
                            
                    #     # plt.scatter(proj[:,0],proj[:,1],c='r')
                    #     cv2.imwrite(f'{self.opt.vis_second_view}/{iter}.jpg', np.concatenate((target, np_img), axis=1) * 255)
                        # import ipdb;ipdb.set_trace()

                        # plt.savefig(os.path.join(self.opt.vis_path,f'test_{iter}.jpg'), proj)
                        # cv2.imwrite(f'./vis_mask/{iter}.jpg', np.concatenate((target_img, masked), axis=1) * 255)
                        
                pbar.set_postfix({'Loss': f'{loss.item():.5f}'})
                # import ipdb;ipdb.set_trace()
                # pbar.set_postfix({'alpha': f'{self.alpha.item():.3f}'})
                # optimize step
                # import ipdb;ipdb.set_trace()
                if epoch > 0 and loss.item() > 0.001:
                    import ipdb;ipdb.set_trace()
                loss.backward()
                overall_loss+=loss.item()
                self.optimizer.step()
                self.writer.add_scalar('Train/Loss', loss.item(), global_step=ep*len(self.dataloader)+iter)   
              
                 
            self.scheduler.step()
            self.writer.add_scalar('Train/Loss Epoch', overall_loss/len(self.dataloader), global_step=ep) 
                

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
            
        

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

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

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
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
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step(i)
                
                
                if i % self.opt.eval_interval == 0:
                    os.makedirs(os.path.join(self.opt.vis_eval_depth_path),exist_ok=True)
                    os.makedirs(os.path.join(self.opt.vis_eval),exist_ok=True)
                    self.eval(i)
                    
                        
                
               
                if i % self.opt.save_interval == 0:
                    # self.save_model(mode='model')
                    os.makedirs(f'./checkpoints/{self.opt.save_path}/',exist_ok=True)
                    os.makedirs(f'./checkpoints/{self.opt.save_path}/{self.opt.save_name}',exist_ok=True)
                    with torch.no_grad():
                        torch.save(self.encoder.state_dict(), f'./checkpoints/{self.opt.save_path}/{self.opt.save_name}/epoch{i}.pth')
        self.writer.close()   
            # do a last prune
            # self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
    