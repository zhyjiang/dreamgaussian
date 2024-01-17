from torch.utils.data import DataLoader, dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import imageio
import cv2
import json
import time
import copy
import torch
from random import sample
from SMPL.smpl import SMPL
import smplx

import random


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0][0]
    max_x, max_y, max_z = bounds[1][0]
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

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2


    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_THuman_batch(img, msk, K, R, T, bounds, image_scaling, white_back=False):

    H, W = img.shape[:2]
    H, W = int(H * image_scaling), int(W * image_scaling)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    K_scale[:2, :3] = K_scale[:2, :3] * image_scaling
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    # img_ray_d = ray_d.copy()
    # img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    real_bounds = np.zeros_like(bounds)
    real_bounds[0] = bounds[0] + 0.05
    real_bounds[1] = bounds[1] - 0.05
    bound_mask = get_bound_2d_mask(real_bounds, K_scale, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0
    if mask_bkgd:
        img[bound_mask != 1] = 0 #1 if white_back else 0

    # rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all

    coord = np.zeros([len(ray_o), 2]).astype(np.int64)
    bkgd_msk = msk

    return img, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk


class HuMManDatasetBatch(Dataset):
    def __init__(self, data_root=None, split='train', multi_person=True, num_instance=300, poses_start=0, poses_interval=3, poses_num=2, image_scaling=1/3, white_back=False, sample_obs_view=True, fix_obs_view=False, resolution=None):
        super(HuMManDatasetBatch, self).__init__()
        
        self.data_root = data_root
        self.smpl_path = 'SMPL/ROMP_SMPL/SMPL_NEUTRAL.pth'
        self.split = split
        self.H_image_scaling = 1024/1920
        self.W_image_scaling = 1024/1080
        self.white_back = white_back
        self.sample_obs_view = sample_obs_view
        self.fix_obs_view = fix_obs_view
        self.camera_view_num =  3 if split == 'train' else 1
        self.image_scaling = image_scaling

        self.poses_start = poses_start # start index 0
        self.poses_interval = poses_interval*2 if split=='train' else 12 # interval 1
        self.poses_num = poses_num # number of used poses 30

        self.multi_person = multi_person
        self.num_instance = num_instance if split=='train' else 4
        humans_data_root = os.path.dirname(data_root)
        self.humans_list = os.path.join(humans_data_root, 'train.txt') if split=='train' else os.path.join(humans_data_root, 'test.txt')
        with open(self.humans_list) as f:
            humans_name = f.readlines()

        self.all_humans = [data_root] if not multi_person else [os.path.join(humans_data_root, x.strip()) for x in humans_name]
        print('num of subjects: ', len(self.all_humans))

        # observation pose and view
        self.obs_pose_index = None
        self.obs_view_index = None
        self.cams_all = []
        for subject_root in self.all_humans:
            camera_file = os.path.join(subject_root, 'cameras.json')
            camera = json.load(open(camera_file))
            self.cams_all.append(camera)
        # observation pose and view
        self.obs_pose_index = None
        self.obs_view_index = None

        self.smpl_model = SMPL(self.smpl_path)
        self.smpl = smplx.create(
            model_path='SMPL',
            model_type='smpl',
            gender='neutral')
        self.big_pose_params = self.big_pose_params()
        # smpl/world coordinate
        t_vertices, t_joints = self.smpl_model(poses=self.big_pose_params['poses'], betas=self.big_pose_params['shapes'])

        self.t_vertices = t_vertices.numpy().astype(np.float32)
       
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

    def get_mask(self, mask_path):
        msk = imageio.imread(mask_path)
        msk[msk!=0]=255
        return msk

    def prepare_smpl_params(self, smpl_path):
        params_ori = np.load(smpl_path)
        params = {}
        params['shapes'] = np.array(params_ori['betas'])
        params['poses'] = np.array(params_ori['body_pose'])
        params['Rh'] = np.array(params_ori['global_orient'])
        params['Th'] = np.array(params_ori['transl'])
        return params

    def prepare_input(self, smpl_path, K,R,T):

        params = self.prepare_smpl_params(smpl_path)

        xyz = self.smpl(
            betas=torch.tensor(params['shapes']).view(1, 10),
            body_pose=torch.tensor(params['poses']).view(1, 23,3),
            global_orient=torch.tensor(params['Rh']).view(1, 1, 3),
            transl=torch.tensor(params['Th']).view((1,3)),
            return_verts=True
        )
        

        xyz = xyz.vertices.numpy()

        T_world2cam = np.eye(4)
        T_world2cam[:3, :3] = R
        T_world2cam[:3, 3:] = T

        # convert 3D points to homogeneous coordinates
        points_3D = xyz[0].T  # (3, N)
        points_homo = np.vstack([points_3D, np.ones((1, points_3D.shape[1]))])  # (4, N)

        
        # transform points to the camera frame
        transformed_points = T_world2cam @ points_homo  # (4, N)
        transformed_points = transformed_points[:3, :]  # (3, N)
        transformed_points = transformed_points.T  # (N, 3)
        
        
        proj = (K  @ transformed_points.T).T
        proj[:,:2] /= proj[:,2:]
        
        world_pelvis = self.smpl_model.h36m_joints_extract(transformed_points[None,...])[:, 14].numpy()
        
    
        
        vertices = transformed_points
        
        vertices -= world_pelvis

        return vertices, params, world_pelvis,proj
    def big_pose_params(self):

        big_pose_params = {}
        big_pose_params['Rh'] = np.eye(3).astype(np.float32)
        big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
        big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
        big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)

        return big_pose_params
    
    def get_bbox(self,mask,new_H,new_W):
        non_zero_pixels = np.nonzero(mask)
        minx,miny = np.min(non_zero_pixels, axis=1)
        maxx,maxy = np.max(non_zero_pixels, axis=1)

        return (maxx,maxy,minx,miny)

    def __getitem__(self, index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """
        instance_idx = index // (self.poses_num * self.camera_view_num) if self.multi_person else 0
        pose_index = ( index % (self.poses_num * self.camera_view_num) ) // self.camera_view_num * self.poses_interval + self.poses_start
        view_index = index % self.camera_view_num

        self.data_root = self.all_humans[instance_idx]
        self.cams = self.cams_all[instance_idx]

        if not os.path.exists(os.path.join(self.data_root, 'kinect_color', 'kinect_'+str(0).zfill(3), str(pose_index).zfill(6)+'.png')):
            arr = os.listdir(os.path.join(self.data_root, 'kinect_color', 'kinect_'+str(0).zfill(3)))
            pose_index = int(random.choice(arr).split('.')[0])

        img_all = []
        K_all = []
        fovx, fovy = [],[]
        mask_list = []
        w2c = []
        img_paths = []
        
        

        # Load image, mask, K, D, R, T
        img_path = os.path.join(self.data_root, 'kinect_color', 'kinect_'+str(view_index).zfill(3), str(pose_index).zfill(6)+'.png')            
        mask_path = os.path.join(self.data_root, 'kinect_mask', 'kinect_'+str(view_index).zfill(3), str(pose_index).zfill(6)+'.png')    

        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        msk = np.array(self.get_mask(mask_path)) / 255.
        img[msk == 0] = 1 if self.white_back else 0
        

        K = np.array(self.cams[f'kinect_color_{str(view_index).zfill(3)}']['K'])
        R = np.array(self.cams[f'kinect_color_{str(view_index).zfill(3)}']['R'])
        T = np.array(self.cams[f'kinect_color_{str(view_index).zfill(3)}']['T']).reshape(-1, 1)

        smpl_path = os.path.join(self.data_root, 'smpl_params', str(pose_index).zfill(6)+'.npz') 
        vertices, params,pelvis,_ = self.prepare_input(smpl_path,K,R,T)
        img = np.transpose(img, (2,0,1))

        # obs view
        fovx = 2 * np.arctan2(1920, 2 *K[0, 0])
        fovy = 2 * np.arctan2(1080, 2 *K[1, 1])
        temp_w2c = np.eye(4).astype(np.float32)
        temp_w2c[:3,3] = pelvis

        ret = {
 
            "smpl_param": params, # smpl params including smpl global R, Th
            'vertices': vertices, #canonical vertices
            'image': img,                                                             
            'mask': msk,
            'K': K,
            'fovx': fovx,
            'fovy': fovy,
            'w2c': temp_w2c,
        
        }

        return ret

    def __len__(self):
        return self.num_instance * self.poses_num * self.camera_view_num


