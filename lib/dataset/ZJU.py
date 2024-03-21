from torch.utils.data.dataset import Dataset
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from SMPL.smpl import SMPL
from lib.gs.gs_renderer import Renderer, MiniCam,Camera
import copy
import math
from torchvision import transforms
from PIL import Image


class ZJU(Dataset):
    def __init__(self, cfg,mode='train'):
        self.cfg = cfg
        self.path = cfg.dataset.path if mode == 'train' else cfg.dataset.test_path
        self.root = cfg.dataset.root
        self.camera_list = cfg.dataset.camera_list
        self.test_camera_list = cfg.dataset.test_camera_list
        self.mode = mode
        self.canonical = cfg.canonical
        self.smpl = SMPL('SMPL/ROMP_SMPL/SMPL_NEUTRAL.pth')
        self.transform = transforms.Compose([
                transforms.Resize((self.cfg.dataset.H, self.cfg.dataset.W)),
            ])
        # self.transform = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #     ])
        self.image_path = []
        
        self.mask_path = []
        for i in range(len(self.camera_list)):
            self.mask_path.append([])
            self.image_path.append([])
        self.smpl_params = []
        self.vertices = []
        self.smpl2w = []
        self.camera_params = []
        self.multi_view = cfg.multi_view
        self.H = cfg.dataset.H
        self.W = cfg.dataset.W
        
        
        for i in range(len(self.camera_list)):
            self.camera_params.append(
        {
            'Ks': [],
            'R': [],
            'T': [],
            'Ths': [],
            'Poses': [],
        })
        self.Ks = []
        self.RTs = []
        self.Ths = []
        self.Poses = []
        
        self.read_data()
        
        self.sample_rate = cfg.dataset.sample_rate if self.mode == 'train' else cfg.dataset.eval_sample_rate
        self.smpl_coord = cfg.dataset.smpl_coord
        
    def process(self, array):
        res = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                res.append(array[i][j])
        return np.array(res)
    
    def read_data(self):
        
        self.seqid = []
        self.sec_img = []
        self.sec_fx  = []
        self.sec_fy = []
        
        for i in range(len(self.path)):
            sub_path= os.path.join(self.root,'CoreView_'+str(self.path[i]))
            print(f'Loading smpl params from {sub_path}')
            self.annots = np.load(os.path.join(sub_path, 'annots.npy'), allow_pickle=True).item()['cams']

            for cam_id, camera in enumerate(self.camera_list):
                
                sub_image_path = self.image_path[cam_id]
                sub_mask_path = self.mask_path[cam_id]
                if self.path[i] in (313, 315):
                    image_folder = os.path.join(sub_path, f'Camera ({camera})')
                    mask_folder = os.path.join(sub_path, 'mask', f'Camera ({camera})')
                    image_paths = sorted(os.listdir(image_folder))
                else:
                    image_folder = os.path.join(sub_path, f'Camera_B{camera}')
                    mask_folder = os.path.join(sub_path, 'mask', f'Camera_B{camera}')
                    image_paths = sorted(os.listdir(image_folder))
                for image_path in image_paths:
                    sub_image_path.append(os.path.join(image_folder, image_path))
                    if cam_id == 0:
                        self.seqid.append(i)
                        
                    mask_path = image_path.replace('jpg', 'png')
                    
                    if self.path[i] not in (313, 315):
                        sub_mask_path.append(os.path.join(sub_path, 'mask', f'Camera_B{camera}', mask_path))
                    else:
                        sub_mask_path.append(os.path.join(sub_path, 'mask', f'Camera ({camera})', mask_path))
                    
                # import ipdb;ipdb.set_trace()
                    # self.mask_path.append(os.path.join(sub_path, 'mask', f'Camera ({camera})', mask_path))
                # import ipdb;ipdb.set_trace()
                # if self.camera_params[cam_id]['Ks'] is None:
                    # self.camera_params[cam_id]['Ks'] = np.array(self.annots['K'][camera - 1])
                # import ipdb;ipdb.set_trace()
                
                # if self.path[i] in (313, 315):
                # import ipdb;ipdb.set_trace()
                
                if camera == 22:
                    self.camera_params[cam_id]['Ks'].append(self.annots['K'][-2])
                    # RT = np.array([self.annots['RT']][-2])
                    # RT = np.concatenate([self.annots['R'][-2], 
                    #                 self.annots['T'][-2]],
                    #                 axis=-1)
                    # RT[:, -1] *= 1e-3
                    # # D = np.array([self.annots['D']][-2])
                    # R = RT[:3,:3]
                    # T = RT[:3,3]
                    # c2w = np.eye(4)
                    # c2w[:3,:3] = R
                    # c2w[:3,3:4] = T.reshape(-1, 1)
                    # w2c = np.linalg.inv(c2w)
                    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    # T = w2c[:3, 3]
                    
                    R = self.annots['R'][-2].T
                    T = self.annots['T'][-2]/1000.0
                    
                    self.camera_params[cam_id]['R'].append(R)
                    self.camera_params[cam_id]['T'].append(T)
                    
                elif camera == 23:
                    self.camera_params[cam_id]['Ks'].append(self.annots['K'][-1])
                    R = self.annots['R'][ -1].T
                    T = self.annots['T'][ -1]/1000.0
                    # RT = np.array([self.annots['RT']][-1])
                    # RT = np.concatenate([self.annots['R'][-1], 
                    #                 self.annots['T'][-1]],
                    #                 axis=-1)
                    # RT[:, -1] *= 1e-3
                    # # D = np.array([self.annots['D']][-1])
                    # R = RT[:3,:3]
                    # T = RT[:3,3]
                    # c2w = np.eye(4)
                    # c2w[:3,:3] = R
                    # c2w[:3,3:4] = T.reshape(-1, 1)
                    # w2c = np.linalg.inv(c2w)
                    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    # T = w2c[:3, 3]
                    
                    self.camera_params[cam_id]['R'].append(R)
                    self.camera_params[cam_id]['T'].append(T)
                else:
                    self.camera_params[cam_id]['Ks'].append(self.annots['K'][camera - 1])
                    R = self.annots['R'][camera - 1].T
                    T = self.annots['T'][camera - 1]/1000.0
                    # RT = np.array([self.annots['RT']][camera - 1])
                    # RT = np.concatenate([self.annots['R'][camera - 1], 
                    #                 self.annots['T'][camera - 1]],
                    #                 axis=-1)
                    # RT[:, -1] *= 1e-3
                    # D = np.array([self.annots['D']][camera - 1])
                    # R = RT[:3,:3]
                    # T = RT[:3,3]
                    # c2w = np.eye(4)
                    # c2w[:3,:3] = R
                    # c2w[:3,3:4] = T.reshape(-1, 1)
                    # w2c = np.linalg.inv(c2w)
                    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    # T = w2c[:3, 3]
                    
                    self.camera_params[cam_id]['R'].append(R)
                    self.camera_params[cam_id]['T'].append(T)
            
                
                
                    # self.camera_params[cam_id]['RTs']=RT
                # self.camera_params[cam_id]['RTs'].append(RT) 
                
            
            num_frame = len(os.listdir(os.path.join(sub_path, 'new_params')))

            for fid in tqdm(range(num_frame)):
                if self.path[i] in (313, 315):
                    fid = fid + 1
                # import ipdb;ipdb.set_trace()
                self.smpl_params.append(np.load(os.path.join(sub_path, 'new_params', f'{fid}.npy'), allow_pickle=True).item())
                vertices = np.load(os.path.join(sub_path, 'new_vertices', f'{fid}.npy'), allow_pickle=True)
                vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=-1)
                self.vertices.append(vertices)
             
        try:
            assert len(self.camera_list)*len(self.image_path[0]) == len(self.camera_list)*len(self.mask_path[0]) == \
                len(self.smpl_params)*len(self.camera_list)  == \
                len(self.vertices) * len(self.camera_list)
        except:
        
            import ipdb;ipdb.set_trace()
     
        self.vertices = np.array(self.vertices)
        self.image_path = np.array(self.image_path)
        self.mask_path = np.array(self.mask_path)
        self.camera_params = np.array(self.camera_params)
        
        # import ipdb;ipdb.set_trace()
            
    def get_bbox(self,mask,new_H,new_W):
        non_zero_pixels = np.nonzero(mask)
        minx,miny = np.min(non_zero_pixels, axis=1)
        maxx,maxy = np.max(non_zero_pixels, axis=1)
      
        return (maxx,maxy,minx,miny)

    def __getitem__(self, index):
       
        w2c = np.stack([np.eye(4).astype(np.float32)]*len(self.camera_list))
        index = index * self.sample_rate
        
        vertices = self.vertices[index % len(self.vertices)]
        smpl_param = self.smpl_params[index % len(self.vertices)]
        
        # import ipdb;ipdb.set_trace() 
        Rh = smpl_param['Rh']
        
        if Rh.shape[-1]!=3 or Rh.shape[-2]!=3:
            
            Rh = cv2.Rodrigues(np.array(smpl_param['Rh']))[0]
        
            smpl_param['Rh'] = Rh   
            
            # import ipdb;ipdb.set_trace()
        
        
        image_list = []
        ori_image_list = []
        mask_list = []
        corners = []
        ori_mask_list = []
        bound_mask_list= []
        ori_bound_mask_list = []
        fovx = []
        fovy = []
        v = []
        K = []
        R = []
        T = []

        for i in range(len(self.camera_list)):
            
            # if self.multi_view > 1:
            
            temp_v = vertices
            
            cur_R = self.camera_params[i]['R'][self.seqid[index]]
            cur_T = self.camera_params[i]['T'][self.seqid[index]]
            cur_K = copy.deepcopy(self.camera_params[i]['Ks'][self.seqid[index]])
            
            
            w2c = np.eye(4)
            w2c[:3,:3] = cur_R.T
            w2c[:3,3:4] = cur_T
            # import ipdb;ipdb.set_trace()
            min_xyz = np.min(temp_v[:,:3], axis=0)
            max_xyz = np.max(temp_v[:,:3], axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)
            
           

            
            # import ipdb;ipdb.set_trace()
            
            
            

                # if self.canonical:
                #     Rh  = cv2.Rodrigues(self.smpl_params[index]['Rh'])[0]
                #     temp_v[...,:3] = ( (temp_v[...,:3]-self.smpl_params[index]['Th']) @ Rh )
                    
                   
                #     w2c[i][:3,:] = self.camera_params[i]['RTs'][self.seqid[index]]
                #     w2c[i][:, 3:] += pelvis.T
   
                # if self.smpl_coord:
                    
                #     w2c[i][:3,:] = self.camera_params[i]['RTs'][self.seqid[index]]
                    
                    # temp = (temp_v @ self.camera_params[i]['RTs'][self.seqid[index]].T)
                    # proj = (self.camera_params[i]['Ks'][self.seqid[index]]  @ temp.T).T
                    # proj[:,:2] /= proj[:,2:]
                    
                    # np.save('proj.npy',proj)
                    # np.save('img.npy',cv2.imread(self.image_path[i][index]).astype(np.float32) / 255)
                    # w2c[i][:, 3:] += pelvis.T
            # else:
            #         temp_v = vertices
                    
            #         w2c[i][:3,:] = self.camera_params[i]['RTs'][self.seqid[index]]
                
            v.append(temp_v)
            
            cur_image = Image.open(self.image_path[i][index])
            # ori_image_list.append(np.array(cur_image))
    
            mask_image = Image.open(self.mask_path[i][index])
            # ori_mask_list.append(np.array(mask_image))
            
            # if self.H != 1024:
            
            #     cur_image = np.array(self.transform(cur_image))/255
            # else:
            #     cur_image = np.array(cur_image)/255
            cur_image = np.array(cur_image)/255

           

            # if self.H != 1024:
            #     mask_image = np.array(self.transform(mask_image))
            # else:
            mask_image = np.array(mask_image)

            
            
            mask_image[mask_image!=0] = mask_image.max()
            # cur_image[mask_image==0] = 0
            # import ipdb;ipdb.set_trace()
            cur_K[:2,:] = cur_K[:2,:] * self.H/1024
            # import ipdb;ipdb.set_trace()
            cur_image *= mask_image[...,None]
            mask_list.append(mask_image)
            image_list.append(cur_image)
            
            bound_mask,max_corner,min_corner = self.get_bound_2d_mask(world_bound, cur_K, w2c[:3],self.H ,self.W,return_corners=True)
            bound_mask = bound_mask.astype(np.float32)
            bound_mask_list.append(bound_mask)
            # cur_image = self.transform(cv2.imread(self.image_path[i][index]).astype(np.float32).transpose(2, 0, 1) / 255)
            # # import ipdb;ipdb.set_trace()
            # mask_image = cv2.imread(self.mask_path[i][index])[:, :, 0]
            # mask_image[mask_image!=0] = mask_image.max()
            # # cur_image[mask_image==0] = 0
            # cur_image *= mask_image[None,:,:]
            # cur_image = self.transform(cur_image)
            # mask_list.append(mask_image)
            # image_list.append(cur_image)
            
            # cur_K[:2,:] = cur_K[:2,:] * 224/ 1024
            # import ipdb;ipdb.set_trace()
            K.append(cur_K)
            R.append(cur_R)
            T.append(cur_T)
            corners.append((max_corner,min_corner))
            
            fovx.append(self.focal2fov(cur_K[0,0], self.cfg.dataset.H))
            fovy.append(self.focal2fov(cur_K[1,1], self.cfg.dataset.W))
            
            

            # fovx.append(2 * np.arctan2(1024, 2 * self.camera_params[i]['Ks'][self.seqid[index]][0, 0]))
            # fovy.append(2 * np.arctan2(1024, 2 * self.camera_params[i]['Ks'][self.seqid[index]][1, 1]))
            
        # import ipdb;ipdb.set_trace()
        return {
            'vertices': np.array(v)[...,:3],
            'image': np.array(image_list),
            'mask': np.array(mask_list),
            'K': np.array(K),
            'R': np.array(R),
            'T': np.array(T),
            'image_path': self.image_path[0][index],
            'smpl_param': smpl_param,
            'fovx': np.array(fovx),
            'fovy': np.array(fovy),
            'bound_mask': np.array(bound_mask_list),
            'corners': np.array(corners),
            # 'w2c': w2c,
        }

    def __len__(self):
        return len(self.image_path[0]) // self.sample_rate
    
    def focal2fov(self,focal, pixels):
        return 2*math.atan(pixels/(2*focal))
    
    
    def get_bound_corners(self,bounds):
        min_x, min_y, min_z = bounds[0]
        max_x, max_y, max_z = bounds[1]
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return corners_3d

    def project(self,xyz, K, RT):
        """
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    def get_bound_2d_mask(self,bounds, K, pose, H, W,return_corners=False):
        # import ipdb;ipdb.set_trace()
        corners_3d = self.get_bound_corners(bounds)
        corners_2d = self.project(corners_3d, K, pose)
        corners_2d = np.round(corners_2d).astype(int)
        # import ipdb;ipdb.set_trace()
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
        cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
        cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
        cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
        cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
        cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
        if return_corners:
            return mask, np.max(corners_2d,axis=0),np.min(corners_2d,axis=0)
        return mask
    
    
    
    
    # def load_map(self,i,info):
    
        

        

    #     return Camera( R=info.R, T=info.T, K=info.K, 
    #                 FoVx=info.FovX, FoVy=info.FovY, 
    #                 data_device=info.data_device)
    
    
