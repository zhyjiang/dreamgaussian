### Train Script

python tools/train.py --config configs/humangaussian/zju.yaml 

make sure you download SMPl from the google zip and unzip in the project root directory. 

Make sure change save_name and all viuslization path name （vis_path, vis_depth_path, vis_eval, vis_eval_depth_path）after each run!!

PLease run these tasks.

1. in the config file set param_input = True cros_attn = True
2. param_input = True cros_attn = False
3. param_input = False cros_attn = True
4. param_input = False cros_attn = False
5. param_input = True cros_attn = True set IOU_weight = 0.02
6. param_input = True cros_attn = True set smpl_reg_scale = 0.1
