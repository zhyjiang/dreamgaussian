### Train Script

python tools/train.py --config configs/humangaussian/zju.yaml 

make sure you download SMPl from the google zip and unzip in the project root directory. 

Make sure change save_name and all viuslization path name （vis_path, vis_depth_path, vis_eval, vis_eval_depth_path）after each run!!

PLease run these tasks.

remember to change dataset2 root to the local humman path in the config file

1. in the config file set full_token = True dino_update = True
2. full_token = False dino_update = True
3. full_token = False dino_update = False
4. full_token = True reshape = True dino_update = False
5. param_input = True cros_attn = True smpl_reg_scale = 0.01


