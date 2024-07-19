import os
import numpy as np
import torch.nn.functional as F
import trimesh

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.bounding_boxes import BoundingBoxes
from typing import Union

from einops import rearrange, repeat
from utils.transformations import transform_body_pose


def render_motion_pair(renderer: Union[HeadlessRenderer, Viewer], datum: dict,
                  filename: str, pose_repr: str,
                  camera_lock_offset: tuple = (2, 2, 2),
                  batch_first=False, only_skel=False,
                  **kwargs) -> None:
    """
    Function to render a video of a motion sequence
    renderer: aitviewer renderer
    datum: dictionary containing sequence of poses, body translations and body orientations
        data could be numpy or pytorch tensors
    filename: the absolute path you want the video to be saved at
    """
    def preproc_motion(m):
        in_format = '1 f d' if batch_first else 'f 1 d'
        assert {'body_transl', 'body_orient', 'body_pose'}.issubset(set(m.keys()))
        body_transl = rearrange(m['body_transl'],f'{in_format} -> f d')
        # make axis-angle
        global_orient = transform_body_pose(m['body_orient'], f"{pose_repr}->aa")
        body_pose = transform_body_pose(m['body_pose'], f"{pose_repr}->aa")
        # remove singleton batch dimention and  flatten axis-angle
        global_orient = rearrange(global_orient, f'{in_format} -> f d')
        body_pose = rearrange(body_pose, f'{in_format} -> f d')
        return {'global_orient': global_orient, 'body_pose': body_pose,
                'body_transl': body_transl}
    
    m1 = preproc_motion(datum['motion1'])
    m2 = preproc_motion(datum['motion2'])

    scene_elements = []
    smpl_template1 = SMPLSequence(m1['body_pose'],
                                 SMPLLayer(model_type='smplx',
                                           num_pca_comps=6,
                                           v_template=None,
                                           gender='neutral',
                                           device=C.device,
                                           ),
                                 poses_root=m1['global_orient'],
                                 trans=m1['body_transl'],
                                 color=(169 / 255, 169 / 255, 169 / 255, 1.0)
                                 )
    smpl_template2 = SMPLSequence(m2['body_pose'],
                                 SMPLLayer(model_type='smplx',
                                           num_pca_comps=6,
                                           v_template=None,
                                           gender='neutral',
                                           device=C.device,
                                           ),
                                 poses_root=m2['global_orient'],
                                 trans=m2['body_transl'],
                                 color=(100 / 255, 149 / 255, 237 / 255, 1.0)
                                 )
    # camera follows the first smpl sequence
    camera = renderer.lock_to_node(smpl_template1, camera_lock_offset, smooth_sigma=5.0)
    if only_skel:
        smpl_template1.remove(smpl_template1.mesh_seq)
        smpl_template2.remove(smpl_template2.mesh_seq)
    renderer.scene.add(smpl_template1)
    renderer.scene.add(smpl_template2)
    scene_elements.append(smpl_template1)
    scene_elements.append(smpl_template2)


    if isinstance(renderer, HeadlessRenderer):
        renderer.save_video(video_dir=filename, output_fps=30, **kwargs)
        # aitviewer adds a counter to the filename, we remove it
        os.rename(filename[:-4] + '_0.mp4', filename[:-4] + '.mp4')

        # empty scene for the next rendering
        renderer.scene.remove(*scene_elements)
        renderer.scene.remove(camera)
        return None
    elif isinstance(renderer, Viewer):
        print('running interactive Viewer')
        renderer.run()

def render_motion(renderer: Union[HeadlessRenderer, Viewer], datum: dict,
                  filename: str, pose_repr: str,
                  camera_lock_offset: tuple = (2, 2, 2),
                  batch_first=False, only_skel=False,
                  visualise_intention: bool=True, **kwargs) -> None:
    """
    Function to render a video of a motion sequence
    renderer: aitviewer renderer
    datum: dictionary containing sequence of poses, body translations and body orientations
        data could be numpy or pytorch tensors
    filename: the absolute path you want the video to be saved at
    """
    in_format = '1 f d' if batch_first else 'f 1 d'
    assert {'body_transl', 'body_orient', 'body_pose'}.issubset(set(datum.keys()))
    body_transl = rearrange(datum['body_transl'],f'{in_format} -> f d')
    # make axis-angle
    global_orient = transform_body_pose(datum['body_orient'], f"{pose_repr}->aa")
    body_pose = transform_body_pose(datum['body_pose'], f"{pose_repr}->aa")
    # remove singleton batch dimention and  flatten axis-angle
    global_orient = rearrange(global_orient, f'{in_format} -> f d')
    body_pose = rearrange(body_pose, f'{in_format} -> f d')

    scene_elements = []

    # use other information that might exist in the datum dictionary
    sbj_vtemp = None
    if 'v_template' in datum.keys():
        sbj_mesh = os.path.join(datum['v_template'])
        sbj_vtemp = np.array(trimesh.load(sbj_mesh).vertices)
    gender = 'neutral'
    if 'gender' in datum.keys():
        gender = datum['gender']
    n_comps = 6  # default value of smplx
    if 'n_comps' in datum.keys():
        n_comps = datum['n_comps']
    if 'joints' in datum.keys():
        joints = datum['joints']
    smpl_template = SMPLSequence(body_pose,
                                 SMPLLayer(model_type='smplx',
                                           num_pca_comps=n_comps,
                                           v_template=sbj_vtemp,
                                           gender=gender,
                                           device=C.device,
                                           # model_path="/home/mdiomataris/models/smplx"
                                           ),
                                 poses_root=global_orient,
                                 trans=body_transl,
                                 # poses_left_hand=lhand_params['fullpose'],
                                 # poses_right_hand=rhand_params['fullpose'],
                                 color=(169 / 255, 169 / 255, 169 / 255, 1.0)
                                 )
    # camera follows smpl sequence
    camera = renderer.lock_to_node(smpl_template, camera_lock_offset, smooth_sigma=5.0)
    if only_skel:
        smpl_template.remove(smpl_template.mesh_seq)
    renderer.scene.add(smpl_template)
    scene_elements.append(smpl_template)

    if 'goal_static' in datum.keys():
        for g in datum['goal_static']:
            goal_sphere = Spheres(repeat(g, 'd -> f d', f=body_pose.shape[0]),
                                  color=(1.0, 0.0, 1.0, 1.0),
                                  radius=0.05)
            renderer.scene.add(goal_sphere)
            scene_elements.append(goal_sphere)
    if 'goal' in datum.keys():
        goal_sphere = Spheres(datum['goal'], color=(1.0, 0.0, 1.0, 1.0),
                            radius=0.05)
        renderer.scene.add(goal_sphere)
        scene_elements.append(goal_sphere)

    if 'right_wrist_intention_pelv_xy' in datum.keys() and visualise_intention:
        intention = rearrange(datum['right_wrist_intention_pelv_xy'],f'{in_format} -> f 1 d')
        assert 'joints' in datum.keys(), "Need joints to visualise intention, include them in the datum"
        arr_start = joints[:, :, 21]
        arr_end =  arr_start + intention
        intention_arrow = Arrows(arr_start,
                                 arr_end,
                                 color=(0.5, 0.0, 1.0, 1.0))
        renderer.scene.add(intention_arrow)
        scene_elements.append(intention_arrow)

    if 'pelvis_transl_intention_pelv_xy' in datum.keys() and visualise_intention:
        intention = rearrange(datum['pelvis_transl_intention_pelv_xy'],f'{in_format} -> f 1 d')
        assert 'joints' in datum.keys(), "Need joints to visualise intention, include them in the datum"
        arr_start = joints[:, :, 0]
        arr_end =  arr_start + intention
        intention_arrow = Arrows(arr_start,
                                 arr_end,
                                 color=(1.0, 0.0, 0.5, 1.0))
        renderer.scene.add(intention_arrow)
        scene_elements.append(intention_arrow)

    if 'pelvis_transl_exp_delta_pelv_xy' in datum.keys() and visualise_intention:
        intention = rearrange(datum['pelvis_transl_exp_delta_pelv_xy'],f'{in_format} -> f 1 d')
        assert 'joints' in datum.keys(), "Need joints to visualise intention, include them in the datum"
        arr_start = joints[:, :, 0]
        arr_end =  arr_start + intention
        intention_arrow = Arrows(arr_start,
                                 arr_end,
                                 color=(1.0, 0.0, 0.5, 1.0))
        renderer.scene.add(intention_arrow)
        scene_elements.append(intention_arrow)

    if 'body_z_orient_exp_delta_intention_pelv_xy' in datum.keys() and visualise_intention:
        intention = rearrange(datum['body_z_orient_exp_delta_intention_pelv_xy'],f'{in_format} -> f 1 d')
        assert 'joints' in datum.keys(), "Need joints to visualise intention, include them in the datum"
        arr_start = joints[:, :, 0]
        arr_end =  arr_start + intention
        intention_arrow = Arrows(arr_start,
                                 arr_end,
                                 color=(1.0, 0.5, 0.5, 1.0))
        renderer.scene.add(intention_arrow)
        scene_elements.append(intention_arrow)

    if 'body_z_orient_intention_pelv_xy' in datum.keys() and visualise_intention:
        intention = rearrange(datum['body_z_orient_intention_pelv_xy'],f'{in_format} -> f 1 d')
        assert 'joints' in datum.keys(), "Need joints to visualise intention, include them in the datum"
        arr_start = joints[:, :, 0]
        arr_end =  arr_start + intention
        intention_arrow = Arrows(arr_start,
                                 arr_end,
                                 color=(1.0, 0.5, 0.5, 1.0))
        renderer.scene.add(intention_arrow)
        scene_elements.append(intention_arrow)

    if 'body_transl_intention' in datum.keys() and visualise_intention:
        transl_int = rearrange(datum['body_transl_intention'],f'{in_format} -> f 1 d')
        arr_start = body_transl[:, None, :]
        arr_end =  arr_start + transl_int,
        intention_arrow = Arrows(body_transl[:, None, :],
                                 arr_end,
                                 color=(1.0, 0.0, 1.0, 1.0))
        # renderer.scene.add(intention_arrow)

    if 'intention' in datum.keys() and visualise_intention:
        transl_int = rearrange(datum['intention'],f'{in_format} -> f 1 d')
        arr_start = body_transl[:, None, :]
        arr_end =  arr_start + transl_int
        intention_arrow = Arrows(body_transl[:, None, :],
                                 arr_end,
                                 color=(1.0, 0.0, 1.0, 1.0))
        renderer.scene.add(intention_arrow)
        scene_elements.append(intention_arrow)

    if 'joint_positions' in datum.keys():
        joint_PC = PointClouds(datum['joint_positions'])
        renderer.scene.add(joint_PC)
        scene_elements.append(joint_PC)

    if 'scene' in datum.keys():
        walls = datum['scene'][0].walls
        if walls.nelement() > 0:
            for i in range(walls.shape[0]):
                w = repeat(walls[i], '... -> f ...', f=walls.shape[0])
                p1 = F.pad(w[:, 0], (0, 1), "constant", value=0)
                p2 = F.pad(w[:, 1], (0, 1), "constant", value=1)
                p1[:, 0] += 0.1
                bboxes = BoundingBoxes.from_min_max_diagonal(
                        p1.cpu().numpy(), p2.cpu().numpy())
                renderer.scene.add(bboxes)
                scene_elements.append(bboxes)

    if isinstance(renderer, HeadlessRenderer):
        renderer.save_video(video_dir=filename, output_fps=30, **kwargs)
        # aitviewer adds a counter to the filename, we remove it
        os.rename(filename[:-4] + '_0.mp4', filename[:-4] + '.mp4')

        # empty scene for the next rendering
        renderer.scene.remove(*scene_elements)
        renderer.scene.remove(camera)
        return None
    elif isinstance(renderer, Viewer):
        print('running interactive Viewer')
        renderer.run()
