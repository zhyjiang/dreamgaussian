from torch.utils.data.dataset import Dataset
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from SMPL.smpl import SMPL


class ZJU(Dataset):
    def __init__(self, cfg,mode='train'):
        self.cfg = cfg
        self.path = cfg.dataset.path if mode == 'train' else cfg.dataset.test_path
        self.root = cfg.dataset.root
        self.camera_list = cfg.dataset.camera_list
        self.mode = mode
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
        
        
        for i in range(len(self.camera_list)):
            self.camera_params.append(
        {
            'Ks': [],
            'RTs': []
        })
        self.Ks = []
        self.RTs = []
        
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
        # import ipdb;ipdb.set_trace()
        
        
        
        for i in range(len(self.path)):
            sub_path= os.path.join(self.root,'CoreView_'+str(self.path[i]))
            print(f'Loading smpl params from {sub_path}')
            self.annots = np.load(os.path.join(sub_path, 'annots.npy'), allow_pickle=True).item()['cams']
        
            for cam_id,camera in enumerate(self.camera_list):
                sub_image_path = self.image_path[cam_id]
                sub_mask_path = self.mask_path[cam_id]
                # image_paths = sorted(os.listdir(os.path.join(sub_path, f'Camera ({camera})')))
                image_paths = sorted(os.listdir(os.path.join(sub_path, f'Camera_B{camera}')))
                for image_path in image_paths:
                    # self.image_path.append(os.path.join(sub_path, f'Camera ({camera})', image_path))
                    sub_image_path.append(os.path.join(sub_path,f'Camera_B{camera}',image_path,))
                    mask_path = image_path.replace('jpg', 'png')
                    sub_mask_path.append(os.path.join(sub_path, 'mask', f'Camera_B{camera}', mask_path))
                    # self.mask_path.append(os.path.join(sub_path, 'mask', f'Camera ({camera})', mask_path))
                # import ipdb;ipdb.set_trace()
                self.camera_params[cam_id]['Ks'].append(np.tile(np.array(self.annots['K'][camera - 1]),(len(image_paths),1,1)))
                # self.camera_params[cam_id]['Ks'].append(np.array(self.annots['K'][camera - 1])) 
                RT = np.concatenate([self.annots['R'][camera - 1], 
                                    self.annots['T'][camera - 1]],
                                    axis=-1)
                RT[:, -1] *= 1e-3
                self.camera_params[cam_id]['RTs'].append(np.tile(RT,(len(image_paths),1,1)))
                
                
        
                
                
                
            # import ipdb;ipdb.set_trace()
        # print('Loading smpl params...')
        
            num_frame = len(os.listdir(os.path.join(sub_path, 'new_params')))
            # sub_vertices = []
            # sub_smpl2w = []
            # sub_smpl_params = []
            for fid in tqdm(range(num_frame)):
                self.smpl_params.append(np.load(os.path.join(sub_path, 'new_params', f'{fid}.npy'), allow_pickle=True).item())
                vertices = np.load(os.path.join(sub_path, 'new_vertices', f'{fid}.npy'), allow_pickle=True)
                vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=-1)
                self.vertices.append(vertices)
                # self.vertices.append(vertices - pelvis[None, ...])
                smpl2w = np.eye(4)
                # smpl2w[3, :3] = pelvis[0]
                self.smpl2w.append(smpl2w)
           
            # import ipdb;ipdb.set_trace()
        try:
            assert len(self.camera_list)*len(self.image_path[0]) == len(self.camera_list)*len(self.mask_path[0]) == \
                len(self.smpl_params)*len(self.camera_list)  == \
                len(self.vertices) * len(self.camera_list)
        except:
        
            import ipdb;ipdb.set_trace()
            
        for i in range(len(self.camera_list)):
            self.camera_params[i]['RTs']= np.concatenate(self.camera_params[i]['RTs'])
            self.camera_params[i]['Ks'] = np.concatenate(self.camera_params[i]['Ks'])
        self.vertices = np.array(self.vertices)
        self.image_path = np.array(self.image_path)
        self.mask_path = np.array(self.mask_path)
        self.camera_params = np.array(self.camera_params)
        
        
        # self.vertices = self.process(self.vertices)
        # self.smpl_params = self.process(self.smpl_params)
        # assert len(self.vertices) % len(self.camera_list) == 0
        # self.camera_params['Ks'] = np.concatenate(self.camera_params['Ks'])
        # self.camera_params['RTs'] = np.concatenate(self.camera_params['RTs'])
        
        
       
            
        
        

    def __getitem__(self, index):
        # index = 0
        w2c = [np.eye(4).astype(np.float32)]*len(self.camera_list)
        index = index * self.sample_rate
        res = []
        
        
        vertices = self.vertices[index % len(self.vertices)]
        # vertices = vertices @ self.camera_params['RTs'][index // len(self.vertices)].T
        
        
        
        
        
        # fovx = 2 * np.arctan2(1024, 2 * self.camera_params['Ks'][index // len(self.vertices)][0, 0])
        # fovy = 2 * np.arctan2(1024, 2 * self.camera_params['Ks'][index // len(self.vertices)][1, 1])
        
        
        image_list = []
        mask_list = []
        fovx = []
        fovy = []
        v = []
        K = []
        
        
        
        
        
        for i in range(len(self.camera_list)):
            try:
                temp_v = (vertices @ self.camera_params[i]['RTs'][index].T)
            except:
                import ipdb;ipdb.set_trace()
            if self.smpl_coord:
                pelvis = self.smpl.h36m_joints_extract(temp_v[None, ...])[:, 14].numpy()
                temp_v -= pelvis
                w2c[i][:3, 3] = pelvis
            v.append(temp_v)
            image_list.append(cv2.imread(self.image_path[i][index]).astype(np.float32).transpose(2, 0, 1) / 255)
            mask_list.append(cv2.imread(self.mask_path[i][index])[:, :, 0])
            K.append(self.camera_params[i]['Ks'][index])
            fovx.append(2 * np.arctan2(1024, 2 * self.camera_params[i]['Ks'][index][0, 0]))
            fovy.append(2 * np.arctan2(1024, 2 * self.camera_params[i]['Ks'][index][1, 1]))
            
        
        return {
            'vertices': np.array(v)[0],
            'image': np.array(image_list)[0],
            'mask': np.array(mask_list)[0],
            'K': np.array(K)[0],
            # 'K': self.camera_params['Ks'][index // len(self.vertices)],
            'fovx': np.array(fovx)[0],
            'fovy': np.array(fovy)[0],
            # 'RT': self.camera_params['RTs'][index // len(self.vertices)] @ self.smpl2w[index],
            'w2c': np.array(w2c)[0],
        }

    def __len__(self):
        return len(self.image_path[0]) // self.sample_rate
