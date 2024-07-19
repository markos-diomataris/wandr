from models.base_motion_prior import Human
import torch
# import wandb
import sys
import os
import numpy as np
from rendering.render_utils import render_motion
from utils.misc import cast_dict_to_numpy, cast_dict_to_tensors


# set global random seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

# download model and check if the download was successful
model_file = './model_weights/wandr.ckpt'
if os.path.exists(model_file):
    print('Model already downloaded')
else:
    cmd = 'wget https://keeper.mpdl.mpg.de/f/9b8b7a2463134e49911f/?dl=1 --max-redirect=2 --trust-server-names && mkdir -p model_weights && mv wandr.ckpt model_weights/'
    os.system(cmd)

# check if the model is loaded correctly
if os.path.exists(model_file):
    print('Model loaded successfully')
else:
    print(f"Could not find model at {model_file}")
    sys.exit(1)

model: Human = Human.load_from_checkpoint(model_file)
model.eval()

init_state = np.load('./deps/init_pose.npz')
init_state = {key: init_state[key] for key in init_state.files}
init_state = cast_dict_to_tensors(init_state, device=model.device)

goal = torch.tensor([-2.5, +2.5, 1.0])[None, :].to(model.device)

motion_duration = 6  # seconds

out = model.rollout(init_state, goal, n_steps=motion_duration * 30, return_smpl_joints=True, angle_format='aa')
out = cast_dict_to_numpy(out)

# RENDER
from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C
# os.system("Xvfb :11 -screen 0 640x480x24 &")
# os.environ['DISPLAY'] = ":11"
# these two lines above might be needed for true headless rendering (without monitor)
# check: https://github.com/eth-ait/aitviewer/issues/10
C.update_conf({"playback_fps": 30,
                "auto_set_floor": False,
                "z_up": True,
                "smplx_models": 'data/body_models'})
renderer = HeadlessRenderer()
render_motion(renderer=renderer, datum=out, filename='output.mp4', pose_repr='6d')

print("Done")
