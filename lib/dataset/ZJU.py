from torch.utils.data.dataset import Dataset
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


class ZJU(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.dataset.path
        self.camera_list = cfg.dataset.camera_list
        
        self.image_path = []
        self.mask_path = []
        self.smpl_params = []
        self.vertices = []
        self.camera_params = {
            'Ks': [],
            'RTs': []
        }
        self.Ks = []
        self.RTs = []
        self.read_data()
        
    def read_data(self):
        self.annots = np.load(os.path.join(self.path, 'annots.npy'), allow_pickle=True).item()['cams']
        
        for camera in self.camera_list:
            image_paths = sorted(os.listdir(os.path.join(self.path, f'Camera ({camera})')))
            for image_path in image_paths:
                self.image_path.append(os.path.join(self.path, f'Camera ({camera})', image_path))
                mask_path = image_path.replace('jpg', 'png')
                self.mask_path.append(os.path.join(self.path, 'mask', f'Camera ({camera})', mask_path))

            self.camera_params['Ks'].append(np.array(self.annots['K'][camera - 1]))
            RT = np.concatenate([self.annots['R'][camera - 1], 
                                 self.annots['T'][camera - 1]],
                                axis=-1)
            RT[:, -1] *= 1e-3
            self.camera_params['RTs'].append(RT)
        
        print('Loading smpl params...')
        num_frame = len(os.listdir(os.path.join(self.path, 'new_params')))
        for fid in tqdm(range(num_frame)):
            self.smpl_params.append(np.load(os.path.join(self.path, 'new_params', f'{fid+1}.npy'), allow_pickle=True).item())
            vertices = np.load(os.path.join(self.path, 'new_vertices', f'{fid+1}.npy'), allow_pickle=True)
            vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=-1)
            self.vertices.append(vertices)
        
        assert len(self.image_path) == len(self.mask_path) == \
               len(self.smpl_params) * len(self.camera_list) == \
               len(self.vertices) * len(self.camera_list)

    def __getitem__(self, index):
        vertices = self.vertices[index % len(self.vertices)]
        vertices = vertices @ self.camera_params['RTs'][index // len(self.vertices)].T
        fovx = np.rad2deg(2 * np.arctan2(1024, 2 * self.camera_params['Ks'][index // len(self.vertices)][0, 0]))
        fovy = np.rad2deg(2 * np.arctan2(1024, 2 * self.camera_params['Ks'][index // len(self.vertices)][1, 1]))
        return {
            'vertices': vertices,
            'image': cv2.imread(self.image_path[index]).astype(np.float32).transpose(2, 0, 1) / 255,
            'mask': cv2.imread(self.mask_path[index])[:, :, 0],
            'K': self.camera_params['Ks'][index // len(self.vertices)],
            'fovx': fovx,
            'fovy': fovy,
        }

    def __len__(self):
        return len(self.image_path)
