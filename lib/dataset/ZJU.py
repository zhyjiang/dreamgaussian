from torch.utils.data.dataset import Dataset
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from SMPL.smpl import SMPL
import copy


class ZJU(Dataset):
    def __init__(self, cfg,mode='train'):
        self.cfg = cfg
        self.path = cfg.dataset.path if mode == 'train' else cfg.dataset.test_path
        self.root = cfg.dataset.root
        self.camera_list = cfg.dataset.camera_list
        self.mode = mode
        self.canonical = cfg.canonical
        self.smpl = SMPL('SMPL/ROMP_SMPL/SMPL_NEUTRAL.pth')
        
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
        
        
        for i in range(len(self.camera_list)):
            self.camera_params.append(
        {
            'Ks': [],
            'RTs': [],
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
                    sub_mask_path.append(os.path.join(sub_path, 'mask', f'Camera_B{camera}', mask_path))
                    
                # import ipdb;ipdb.set_trace()
                    # self.mask_path.append(os.path.join(sub_path, 'mask', f'Camera ({camera})', mask_path))
                # import ipdb;ipdb.set_trace()
                # if self.camera_params[cam_id]['Ks'] is None:
                    # self.camera_params[cam_id]['Ks'] = np.array(self.annots['K'][camera - 1])
                # import ipdb;ipdb.set_trace()
                self.camera_params[cam_id]['Ks'].append(np.array(self.annots['K'][camera - 1]))
            
                
                RT = np.concatenate([self.annots['R'][camera - 1], 
                                    self.annots['T'][camera - 1]],
                                    axis=-1)
                RT[:, -1] *= 1e-3
                    # self.camera_params[cam_id]['RTs']=RT
                self.camera_params[cam_id]['RTs'].append(RT) 
                
            
            num_frame = len(os.listdir(os.path.join(sub_path, 'new_params')))

            for fid in tqdm(range(num_frame)):
                if self.path[i] in (313, 315):
                    fid = fid + 1
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
        
        image_list = []
        mask_list = []
        fovx = []
        fovy = []
        v = []
        K = []

        for i in range(len(self.camera_list)):
            
            if self.multi_view > 1:
                temp_v = vertices

                # if self.canonical:
                #     Rh  = cv2.Rodrigues(self.smpl_params[index]['Rh'])[0]
                #     temp_v[...,:3] = ( (temp_v[...,:3]-self.smpl_params[index]['Th']) @ Rh )
                    
                   
                #     w2c[i][:3,:] = self.camera_params[i]['RTs'][self.seqid[index]]
                #     w2c[i][:, 3:] += pelvis.T
   
                if self.smpl_coord:
                    
                    w2c[i][:3,:] = self.camera_params[i]['RTs'][self.seqid[index]]
                    
                    # temp = (temp_v @ self.camera_params[i]['RTs'][self.seqid[index]].T)
                    # proj = (self.camera_params[i]['Ks'][self.seqid[index]]  @ temp.T).T
                    # proj[:,:2] /= proj[:,2:]
                    
                    # np.save('proj.npy',proj)
                    # np.save('img.npy',cv2.imread(self.image_path[i][index]).astype(np.float32) / 255)
                    # w2c[i][:, 3:] += pelvis.T
            else:
                    temp_v = vertices
                    
                    w2c[i][:3,:] = self.camera_params[i]['RTs'][self.seqid[index]]
                
            v.append(temp_v)
            image_list.append(cv2.imread(self.image_path[i][index]).astype(np.float32).transpose(2, 0, 1) / 255)
            mask_image = cv2.imread(self.mask_path[i][index])[:, :, 0]

            mask_list.append(mask_image)
            K.append(self.camera_params[i]['Ks'][self.seqid[index]])
            fovx.append(2 * np.arctan2(1024, 2 * self.camera_params[i]['Ks'][self.seqid[index]][0, 0]))
            fovy.append(2 * np.arctan2(1024, 2 * self.camera_params[i]['Ks'][self.seqid[index]][1, 1]))
              
        return {
            'vertices': np.array(v)[...,:3],
            'image': np.array(image_list),
            'mask': np.array(mask_list),
            'K': np.array(K),
            'smpl_param': smpl_param,
            'fovx': np.array(fovx),
            'fovy': np.array(fovy),
            'w2c': w2c,
        }

    def __len__(self):
        return len(self.image_path[0]) // self.sample_rate
