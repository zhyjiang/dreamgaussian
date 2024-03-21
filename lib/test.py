from data_utils.create_UV_maps import UV_Map_Generator
import cv2
import torch
import numpy as np
from SMPL.smpl_torch import SMPLModel
file_prefix = 'template'

generator = UV_Map_Generator(
    UV_height=224,
    UV_pickle='template.pickle'
)
# uv_map = np.load('x.npy')
uv_map = cv2.imread("/home/zhongyuj/dreamgaussian/UV_ZJU_output_smpl_feature_ss_1.0/uv_map_epoch108_view0.png")
uv_map = torch.tensor(uv_map).cuda().float()/255.0
# import ipdb;ipdb.set_trace()
resample_mesh = generator.resample(uv_map)

# import ipdb;ipdb.set_trace()

device = torch.device('cuda')

model = SMPLModel(
    device=device,
    model_path='/home/zhongyuj/dreamgaussian/lib/model_lsp.pkl',
)

model.write_obj(resample_mesh, './test.obj')
# generator.UV_interp(np.load('m.npy'))