import logging
import random
from glob import glob
from os import listdir
from os.path import exists, join
from hydra.utils import instantiate
import re
from typing import List, Union, Tuple

import joblib
import numpy as np
import smplx
import torch
from einops import rearrange, repeat
from utils.pytorch3d_transforms import matrix_to_euler_angles, matrix_to_rotation_6d
from pytorch_lightning import LightningDataModule
from smplx.joint_names import JOINT_NAMES
from torch.nn.functional import pad
from torch.quantization.observer import \
    MovingAveragePerChannelMinMaxObserver as mmo
from torch.utils.data import DataLoader, Dataset
from utils.masking import LengthMask
from utils.misc import DotDict, cast_dict_to_tensors, to_tensor
from utils.transformations import (
    change_for, local_to_global_orient, transform_body_pose, remove_z_rot,
    rot_diff, get_z_rot, root2heading)
from dataset.feature_functions import (
    body_z_orient_intention_pelv_xy, joint_intention_pelv_xy, full_local_intention,
    joint_exp_distance, joint_intention_pelv_xy_exp_z
    )

from dataset.sequence_parser_amass import SequenceParserAmass
from motion_filter.amass_filter import AmassFilter

# A logger for this file
log = logging.getLogger(__name__)


class AmassDataset(Dataset):
    def __init__(self, data: list, smplx_models_path: str,
                 sequence_parser: SequenceParserAmass, load_feats, statistics_path: str=None,
                 rot_repr="6d", norm_type="6d", do_augmentations=False, **kwargs):
        self.data = data
        self.rot_repr = rot_repr
        self.norm_type = norm_type
        self.load_feats = load_feats
        self.do_augmentations = do_augmentations
        self.seq_parser = sequence_parser
        bm = smplx.create(model_path=smplx_models_path, model_type='smplx')
        self.body_chain = bm.parents
        self.stats = None
        self.joint_idx = {name: i for i, name in enumerate(JOINT_NAMES)}

        if statistics_path is not None and exists(statistics_path):
            stats = np.load(statistics_path, allow_pickle=True)[()]
            self.stats = cast_dict_to_tensors(stats)
        self.kwargs = DotDict(kwargs)

        # declare functions that implement features here
        self._feat_get_methods = {
            "body_transl": self._get_body_transl,
            "body_transl_z": self._get_body_transl_z,
            "body_transl_delta": self._get_body_transl_delta,
            "body_transl_delta_pelv": self._get_body_transl_delta_pelv,
            "body_transl_delta_pelv_xy": self._get_body_transl_delta_pelv_xy,

            "body_transl_intention": self._get_body_transl_intention,
            "body_z_orient_intention": self._get_body_z_orient_intention,
            "body_z_orient_intention_pelv_xy": self._get_body_z_orient_intention_pelv_xy,
            "body_z_orient_intention_pelv_xy_timeless": self._get_body_z_orient_intention_pelv_xy_timeless,
            "body_transl_intention_goal": self._get_body_transl_intention_goal,
            "right_wrist_intention_goal": self._get_right_wrist_intention_goal,
            "rwrist_full_local_intention": self._get_rwrist_full_local_intention,
            "body_transl_intention_pelv_xy": self._get_body_transl_intention_pelv_xy,
            "pelvis_transl_intention_pelv_xy": self._get_pelvis_transl_intention_pelv_xy,
            "right_wrist_intention_pelv_xy": self._get_right_wrist_intention_pelv_xy,
            "right_wrist_intention_pelv_xy_exp_z": self._get_right_wrist_intention_pelv_xy_exp_z,
            "right_wrist_intention": self._get_right_wrist_intention,
            "right_wrist_exp_distance": self._get_right_wrist_exp_distance,
            "pelvis_xy_exp_distance": self._get_pelvis_xy_exp_distance,

            "intention_lookahead_frames": self._get_intention_lookahead_frames,
            "body_orient": self._get_body_orient,
            "body_orient_xy": self._get_body_orient_xy,
            "body_orient_delta": self._get_body_orient_delta,
            "body_pose": self._get_body_pose,
            "body_pose_delta": self._get_body_pose_delta,
            "intention_goal_frame": self._get_intention_goal_frame,
            "plan_window": self._get_plan_window,

            "body_joints": self._get_body_joints,
            "body_joints_rel": self._get_body_joints_rel,
            "body_joints_vel": self._get_body_joints_vel,
            "joint_global_oris": self._get_joint_global_orientations,
            "joint_ang_vel": self._get_joint_angular_velocity,
            "wrists_ang_vel": self._get_wrists_angular_velocity,
            "wrists_ang_vel_euler": self._get_wrists_angular_velocity_euler,
            "goal": self._goal,
            "motion_completion_ratio": self._motion_completion_ratio,
        }

        # declare functions that return metadata here 
        self._meta_data_get_methods = {
            "n_frames_orig": self._get_num_frames,
            "chunk_start": self._get_chunk_start,
            "framerate": self._get_framerate,
        }

    def get_features_dimentionality(self):
        # get the feature dimentionality
        item = self.__getitem__(0)
        return {feat: item[feat].shape[-1] for feat in self.load_feats
                        if feat in self._feat_get_methods.keys()}

    def normalize_feats(self, feats, feats_name):
        if feats_name not in self.stats.keys():
            log.error(f"Tried to normalise {feats_name} but did not found stats \
                      for this feature. Try running calculate_statistics.py again.")
        if self.norm_type == "std":
            mean, std = (self.stats[feats_name]['mean'].to(feats.device),
                         self.stats[feats_name]['std'].to(feats.device))
            return (feats - mean) / (std + 1e-5)
        elif self.norm_type == "norm":
            max, min = (self.stats[feats_name]['max'].to(feats.device),
                        self.stats[feats_name]['min'].to(feats.device))
            return (feats - min) / (max - min + 1e-5)

    def _get_body_joints(self, data):
        joints = to_tensor(data['joint_positions'][:, :22, :])
        return rearrange(joints, '... joints dims -> ... (joints dims)')

    def _get_joint_global_orientations(self, data):
        body_pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        body_orient = to_tensor(data['rots'][..., :3])
        joint_glob_oris = local_to_global_orient(body_orient, body_pose,
                                                 self.body_chain,
                                                 input_format='aa',
                                                 output_format="rotmat")
        return rearrange(joint_glob_oris, '... j k d -> ... (j k d)')

    def _get_joint_angular_velocity(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity_euler(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = transform_body_pose(to_tensor(pose[..., 19:21, :]), "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_euler_angles(rot_diffs, "XYZ"), '... j c -> ... (j c)')

    def _get_body_joints_vel(self, data):
        joints = to_tensor(data['joint_positions'][:, :22, :])
        joint_vel = joints - joints.roll(1, 0)  # shift one right and subtract
        joint_vel[0] = 0
        return rearrange(joint_vel, '... j c -> ... (j c)')

    def _get_body_joints_rel(self, data):
        """get body joint coordinates relative to the pelvis"""
        joints = to_tensor(data['joint_positions'][:, :22, :])
        pelvis_transl = to_tensor(joints[:, 0, :])
        joints_glob = to_tensor(joints[:, :22, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient = transform_body_pose(pelvis_orient, "aa->rot")
        # relative_joints = R.T @ (p_global - pelvis_translation)
        rel_joints = torch.einsum('fdi,fjd->fji',
                                  pelvis_orient, joints_glob - pelvis_transl[:, None, :])
        return rearrange(rel_joints, '... j c -> ... (j c)')

    @staticmethod
    def _get_framerate(data):
        """get framerate"""
        return torch.tensor([data['fps']])

    def _get_plan_window(self, data):
        return self.picking_plan_logic(data)

    def picking_plan_logic(self, data):
        frames = to_tensor(data['joint_positions']).shape[0]
        sample_weights = torch.tril(torch.rand((frames, frames)))
        sample_weights.fill_diagonal_(1e-5)
        plan_start = sample_weights.argmax(1).float()
        sample_weights = torch.tril(torch.rand((frames, frames))).T
        sample_weights.fill_diagonal_(1e-5)
        plan_end = sample_weights.argmax(1).float()
        # stack start and end
        plan_window = torch.stack((plan_start,
                                   torch.arange(frames),
                                   plan_end), dim=-1)
        return plan_window

    def picking_goal_logic(self, data):
        pelvis_joints = to_tensor(data['joint_positions'])[:, 0, :]
        frames = pelvis_joints.shape[0]
        if self.kwargs.picking_goal_logic.mode == 'fixed_offset':
            goal_frame = torch.arange(frames).float()
            offset = self.kwargs.picking_goal_logic.lookahead_frames
            goal_frame[:-offset] += offset
        elif self.kwargs.picking_goal_logic.mode == 'pick_any':
            sample_weights = torch.tril(torch.rand((frames, frames))).T
            goal_frame = sample_weights.argmax(1).float()
        elif self.kwargs.picking_goal_logic.mode == 'use_goal_if_exists_else_any':
            if 'goal_frame' in data.keys() and data['target_bone'] == 'RightHand':
                # we know when the target is reached
                goal_frame = torch.arange(frames).float()
                gf = data['goal_frame']
                if gf == frames:
                    gf -= 1
                goal_frame[:gf] = gf
            else:
                sample_weights = torch.tril(torch.rand((frames, frames))).T
                goal_frame = sample_weights.argmax(1).float()
        elif self.kwargs.picking_goal_logic.mode == 'only_use_goal_when_close':
            if 'goal_frame' in data.keys() and data['target_bone'] == 'RightHand':
                # we know when the target is reached
                goal_frame = torch.arange(frames).float()
                gf = data['goal_frame']
                if gf == frames:
                    gf -= 1
                goal_frame[:gf] = gf
            else:
                pdist = torch.cdist(pelvis_joints[..., :2], pelvis_joints[..., :2])
                sample_weights = (pdist >= self.kwargs.picking_goal_logic.cutoff_distance).float() 
                sample_weights *= torch.rand_like(sample_weights)
                # add some small probability to the diagonal in case there are no valid targets
                sample_weights += torch.eye(sample_weights.shape[0], sample_weights.shape[1]) * 0.01
                sample_weights *= torch.tril(torch.ones_like(sample_weights)).T
                # multiply by lower triangular
                goal_frame = sample_weights.argmax(1).float()
        else:
           raise ValueError(f'picking_goal_logic.mode is unknown: {self.kwargs.picking_goal_logic.mode}') 
        goal_frame = rearrange(goal_frame, 'f -> f 1')
        return goal_frame

    def _get_intention_goal_frame(self, data):
        return self.picking_goal_logic(data)

    def _get_intention_lookahead_frames(self, data):
        """get lookahead frames of goals"""
        trans = to_tensor(data['trans'])
        lookahead = torch.ones_like(trans[:, :1])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        lookahead *= k
        return lookahead

    @staticmethod
    def _get_chunk_start(data):
        """get number of original sequence frames"""
        return torch.tensor([data['chunk_start']])

    @staticmethod
    def _get_num_frames(data):
        """get number of original sequence frames"""
        return torch.tensor([data['rots'].shape[0]])

    def _get_body_transl(self, data):
        """get body pelvis tranlation"""
        return to_tensor(data['trans'])

    def _motion_completion_ratio(self, data):
        """get body pelvis tranlation"""
        return torch.arange(data['rots'].shape[0]) / (data['rots'].shape[0] - 1)

    def _get_body_transl_z(self, data):
        """get body pelvis tranlation"""
        return to_tensor(data['trans'])[..., 2]

    def _get_body_transl_delta(self, data):
        """get body pelvis tranlation delta"""
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        trans_vel[0] = 0  # zero out velocity of first frame
        return trans_vel

    def _get_body_transl_delta_pelv(self, data):
        """
        get body pelvis tranlation delta relative to pelvis coord.frame
        v_i = t_i - t_{i-1} relative to R_{i-1}
        """
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        pelvis_orient =transform_body_pose(to_tensor(data['rots'][..., :3]), "aa->rot")
        trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv

    def _get_right_wrist_intention_goal(self, data):
        """
        get right wrist intention global position
        """
        rwrist = to_tensor(data['joint_positions'][:, 21, :])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        rwrist_int_goal = rwrist.roll(-k, 0)
        rwrist_int_goal[-k:] = rwrist[-k:]  # zero out intention of last frames
        return rwrist_int_goal

    def _get_body_transl_intention_goal(self, data):
        """
        get body tranlation intention global position
        """
        trans = to_tensor(data['trans'])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        trans_int_goal = trans.roll(-k, 0)
        trans_int_goal[-k:] = trans[-k:]  # zero out intention of last frames
        return trans_int_goal

    def _get_body_z_orient_intention(self, data):
        """
        get body orientation intention while 
        """
        pelvis_orient = to_tensor(data['rots'][..., :3])
        heading_vec = root2heading(pelvis_orient)
        k = self.kwargs.picking_goal_logic.lookahead_frames
        heading_int = heading_vec.roll(-k, 0) - heading_vec
        heading_int[-k:] = 0  # zero out intention of last frames
        return heading_int

    def _get_rwrist_full_local_intention(self, data):
        k = self.kwargs.picking_goal_logic.lookahead_frames
        pelvis_orient = to_tensor(data['rots'][..., :3])
        joints = to_tensor(data['joint_positions'])
        rwrist = joints[:, 21, :]
        goal_loc = rwrist.roll(-k, 0)
        heading_goal = goal_loc - joints[:, 0]
        heading_goal[..., 2] = 0
        intention = full_local_intention(pelvis_orient=pelvis_orient,
                                                    joint=rwrist,
                                                    joint_goal=goal_loc,
                                                    heading_goal=heading_goal,
                                                    k=k,
                                                    in_format='aa')
        intention[-k:] = 0  # zero out intention of last frames
        return intention

    def _get_body_z_orient_intention_pelv_xy_timeless(self, data):
        """
        get body orientation intention while removing the global z rotation of the pelvis
        """
        pelvis_orient = to_tensor(data['rots'][..., :3])
        heading_vec = root2heading(pelvis_orient)
        k = self.kwargs.picking_goal_logic.lookahead_frames
        heading_int_pelv = body_z_orient_intention_pelv_xy(
            pelvis_orient, heading_vec.roll(-k, 0), k)
        heading_int_pelv[-k:] = 0  # zero out intention of last frames
        return heading_int_pelv

    def _get_body_z_orient_intention_pelv_xy(self, data):
        """
        get body orientation intention while removing the global z rotation of the pelvis
        """
        pelvis_orient = to_tensor(data['rots'][..., :3])
        heading_vec = root2heading(pelvis_orient)
        k = self.kwargs.picking_goal_logic.lookahead_frames
        heading_int_pelv = body_z_orient_intention_pelv_xy(
            pelvis_orient, heading_vec.roll(-k, 0), k)
        heading_int_pelv[-k:] = 0  # zero out intention of last frames
        return heading_int_pelv

    def _get_body_transl_intention(self, data):
        """
        get body tranlation intention while removing the global z rotation of the pelvis
        v_i = t_{i+k} - t_i relative to R_i_xy
        """
        trans = to_tensor(data['trans'])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        trans_int = trans.roll(-k, 0) - trans
        trans_int[-k:] = 0  # zero out intention of last frames
        trans_int /= (k/30)  # Dx/Dt
        trans_int[..., 2] = 0  # zero out z intention
        return trans_int

    def _get_pelvis_xy_exp_distance(self, data):
        """
        """
        rwrist = to_tensor(data['joint_positions'][:, 0, :])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        exp_dist = joint_exp_distance(rwrist, rwrist.roll(-k, 0), scale=3.0)
        exp_dist[-k:] = 0  # zero out intention of last frames
        return exp_dist

    def _get_right_wrist_exp_distance(self, data):
        """
        """
        rwrist = to_tensor(data['joint_positions'][:, 21, :])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        exp_dist = joint_exp_distance(rwrist, rwrist.roll(-k, 0), scale=5.0)
        exp_dist[-k:] = 0  # zero out intention of last frames
        return exp_dist

    def _get_right_wrist_intention(self, data):
        """
        get body tranlation intention while removing the global z rotation of the pelvis
        v_i = t_{i+k} - t_i relative to R_i_xy
        """
        rwrist = to_tensor(data['joint_positions'][:, 21, :])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        rwrist_int = rwrist.roll(-k, 0) - rwrist
        rwrist_int[-k:] = 0  # zero out intention of last frames
        rwrist_int /= (k/30)  # Dx/Dt
        # rwrist_int_norm = torch.nn.functional.normalize(rwrist_int, dim=1)
        # rwrist_int_scale = rwrist_int.norm(dim=1, keepdim=True)
        return rwrist_int
        # return torch.cat((rwrist_int_int_pelv_norm, rwrist_int_int_pelv_scale), dim=-1)

    def _get_right_wrist_intention_pelv_xy_exp_z(self, data):
        """
        get body tranlation intention while removing the global z rotation of the pelvis
        v_i = t_{i+k} - t_i relative to R_i_xy
        """
        rwrist = to_tensor(data['joint_positions'][:, 21, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        joint_int_pelv = joint_intention_pelv_xy_exp_z(rwrist, rwrist.roll(-k, 0),
                                                       pelvis_orient, k, scale=5.0)
        joint_int_pelv[-k:] = 0  # zero out intention of last frames
        return joint_int_pelv

    def _get_right_wrist_intention_pelv_xy(self, data):
        """
        get body tranlation intention while removing the global z rotation of the pelvis
        v_i = t_{i+k} - t_i relative to R_i_xy
        """
        rwrist = to_tensor(data['joint_positions'][:, 21, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        joint_int_pelv = joint_intention_pelv_xy(rwrist, rwrist.roll(-k, 0),
                                                       pelvis_orient, k)
        joint_int_pelv[-k:] = 0  # zero out intention of last frames
        return joint_int_pelv

    def _get_pelvis_transl_intention_pelv_xy(self, data):
        """
        get body tranlation intention while removing the global z rotation of the pelvis
        v_i = t_{i+k} - t_i relative to R_i_xy
        """
        trans = to_tensor(data['joint_positions'][:, 0, :])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        trans_int = trans.roll(-k, 0) - trans
        trans_int[..., 2] = 0  # zero out z intention
        pelvis_orient = to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_int_pelv = change_for(trans_int, R_z, forward=True)
        trans_int_pelv[-k:] = 0  # zero out intention of last frames
        trans_int_pelv /= (k/30)  # Dx/Dt
        # trans_int_pelv_norm = torch.nn.functional.normalize(trans_int_pelv, dim=1)
        # trans_int_pelv_scale = trans_int_pelv.norm(dim=1, keepdim=True)
        return trans_int_pelv
        # return torch.cat((trans_int_pelv_norm, trans_int_pelv_scale), dim=-1)

    def _get_body_transl_intention_pelv_xy(self, data):
        """
        get body tranlation intention while removing the global z rotation of the pelvis
        v_i = t_{i+k} - t_i relative to R_i_xy
        """
        trans = to_tensor(data['trans'])
        k = self.kwargs.picking_goal_logic.lookahead_frames
        trans_int = trans.roll(-k, 0) - trans
        trans_int[..., 2] = 0  # zero out z intention
        pelvis_orient = to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_int_pelv = change_for(trans_int, R_z, forward=True)
        trans_int_pelv[-k:] = 0  # zero out intention of last frames
        trans_int_pelv /= (k/30)  # Dx/Dt
        # trans_int_pelv_norm = torch.nn.functional.normalize(trans_int_pelv, dim=1)
        # trans_int_pelv_scale = trans_int_pelv.norm(dim=1, keepdim=True)
        return trans_int_pelv
        # return torch.cat((trans_int_pelv_norm, trans_int_pelv_scale), dim=-1)

    def _get_body_transl_delta_pelv_xy(self, data):
        """
        get body pelvis tranlation delta while removing the global z rotation of the pelvis
        v_i = t_i - t_{i-1} relative to R_{i-1}_xy
        """
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        pelvis_orient = to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_vel_pelv = change_for(trans_vel, R_z.roll(1, 0), forward=True)
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv

    def _get_body_orient(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
        return pelvis_orient

    def _get_body_orient_xy(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient_xy = remove_z_rot(pelvis_orient, in_format="aa")
        return pelvis_orient_xy

    def _get_body_orient_delta(self, data):
        """get global body orientation delta"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient_delta = rot_diff(pelvis_orient, in_format="aa", out_format=self.rot_repr)
        return pelvis_orient_delta

    def _get_body_pose(self, data):
        """get body pose"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
        pose = transform_body_pose(pose, f"aa->{self.rot_repr}")
        return pose

    def _get_body_pose_delta(self, data):
        """get body pose rotational deltas"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
        pose_diffs = rot_diff(pose, in_format="aa", out_format=self.rot_repr)
        return pose_diffs

    def _goal(self, data):
        frames = data['trans'].shape[0]
        # sample angle
        rnd = torch.rand(1)
        theta = (rnd * self.kwargs.goal.max_rads + (1 - rnd) * self.kwargs.goal.min_rads) * torch.pi
        rnd = torch.rand(1)
        radious = rnd * self.kwargs.goal.max_radious + (1 - rnd) * self.kwargs.goal.min_radious
        p = torch.tensor([torch.cos(theta), torch.sin(theta), 0]) * radious
        # sample height
        rnd = torch.rand(1)
        p[2] = rnd * self.kwargs.goal.max_height + (1 - rnd) * self.kwargs.goal.min_height
        p = repeat(p, 'd -> f d', f=frames).clone()
        joints = to_tensor(data['joint_positions'][:, :22, :])
        # joints = rearrange(joints, '... joints dims -> ... (joints dims)')
        p[:, :2] += joints[:, 0, :2]
        return p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # parse data: apply augmentations, if any, and add metadata fields
        datum = self.seq_parser.parse_datum(self.data[idx])
        # perform augmentations except when in test mode
        if self.do_augmentations:
            datum = self.seq_parser.augment_npz(datum)

        # compute all features declared in config
        data_dict = {feat: self._feat_get_methods[feat](datum)
                     for feat in self.load_feats}

        # compute their normalised versions as "feat_norm"
        if self.stats is not None:
            norm_feats = {f"{feat}_norm": self.normalize_feats(data, feat)
                        for feat, data in data_dict.items()
                        if feat in self.stats.keys()}
            # append normalized features to data_dict
            data_dict = {**data_dict, **norm_feats}
        # add some meta-data
        meta_data_dict = {feat: method(datum)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict, **meta_data_dict}
        data_dict['filename'] = datum['fname']
        data_dict['split'] = datum['split']
        data_dict['id'] = datum['id']
        return DotDict(data_dict)

    def npz2feats(self, idx, npz):
        """turn npz data to a proper features dict"""
        data_dict = {feat: self._feat_get_methods[feat](npz)
                     for feat in self.load_feats}
        if self.stats is not None:
            norm_feats = {f"{feat}_norm": self.normalize_feats(data, feat)
                        for feat, data in data_dict.items()
                        if feat in self.stats.keys()}
            data_dict = {**data_dict, **norm_feats}
        meta_data_dict = {feat: method(npz)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict, **meta_data_dict}
        data_dict['filename'] = self.file_list[idx]['filename']
        data_dict['split'] = self.file_list[idx]['split']
        return DotDict(data_dict)

    def get_all_features(self, idx):
        datum = self.data[idx]

        data_dict = {feat: self._feat_get_methods[feat](datum)
                     for feat in self._feat_get_methods.keys()}
        return DotDict(data_dict)


class AmassDataModule(LightningDataModule):

    def __init__(self, load_files: str, batch_size: int, num_workers: int,
                 dataset_cfg,  split_seed: int, motion_filter=None,
                 use_debug_dataset=False,
                 shuffle_datasets: List[bool]=[True, True, False], **kwargs):
        super().__init__()
        self.load_files = load_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed
        self.dataset_cfg = dataset_cfg
        self.motion_filter = motion_filter
        self.use_debug_dataset = use_debug_dataset
        self.shuffle_datasets = shuffle_datasets
        self.kwargs = DotDict(kwargs)
        self.dataset = {}

    def setup(self, stage):
        # load data
        if self.use_debug_dataset and 'debug_files' in self.kwargs.keys():
            ds_db_paths = self.kwargs['debug_files']
        else:
            ds_db_paths = self.load_files
        data_dict = {}
        for path in ds_db_paths: 
            log.info(f"Loading {path}")
            new_data_dict = cast_dict_to_tensors(joblib.load(path))
            log.info(f"Loaded {path}")
            if len(iter(new_data_dict.values()).__next__()['rots'].shape) == 3:
                for k, v in new_data_dict.items():
                    new_data_dict[k]['rots'] = rearrange(v['rots'], 'f j d -> f (j d)')

            # add dataset information
            dataset_name = ''
            if path.endswith('circle.pth.tar'):
                dataset_name = 'circle'
            elif path.endswith('circle_small.pth.tar'):
                dataset_name = 'circle'
            elif path.endswith('GRAB.pth.tar'):
                dataset_name = 'grab'
            elif path.endswith('amass_small.pth.tar'):
                dataset_name = 'amass'
            elif path.endswith('amass.pth.tar'):
                dataset_name = 'amass'
            for k, v in new_data_dict.items():
                new_data_dict[k]['dataset_name'] = dataset_name
            
            data_dict = new_data_dict | data_dict
        motion_filter = instantiate(self.motion_filter)
        
        # filter data
        initial_len = len(data_dict)
        if motion_filter is not None:
            data_dict = {k: v for k, v in data_dict.items()
                         if motion_filter.is_good_quality(v)}
        print(f"Filtered {initial_len - len(data_dict)} out of {initial_len} sequences")
                
        # calculate splits
        random.seed(self.split_seed)
        data_ids = list(data_dict.keys())
        data_ids.sort()
        random.shuffle(data_ids)
        # 80-10-10% train-val-test for each sequence
        num_train = int(len(data_ids) * 0.8)
        num_val = int(len(data_ids) * 0.1)
        # give ids to data sets--> 0:train, 1:val, 2:test
        split = np.zeros(len(data_ids), dtype=np.int64)
        split[num_train:num_train + num_val] = 1
        split[num_train + num_val:] = 2
        id_split_dict = {id: split[i] for i, id in enumerate(data_ids)}
        random.random()  # restore randomness in life (maybe randomness is life)

        # setup collate function meta parameters
        self.collate_fn = lambda b: collate_batch(b, self.dataset_cfg.load_feats)
        for k, v in data_dict.items():
            v['id'] = k
            v['split'] = id_split_dict[k]

        # create datasets
        self.dataset['train'], self.dataset['val'], self.dataset['test'] = (
           instantiate(self.dataset_cfg,
                       data=[v for k, v in data_dict.items() if id_split_dict[k] == 0],
                       do_augmentations=True), 
           instantiate(self.dataset_cfg,
                       data=[v for k, v in data_dict.items() if id_split_dict[k] == 1],
                       do_augmentations=True), 
           instantiate(self.dataset_cfg,
                       data=[v for k, v in data_dict.items() if id_split_dict[k] == 2],
                       do_augmentations=False), 
        )
        for splt in ['train', 'val', 'test']:
            log.info("Set up {} set with {} items."\
                     .format(splt, len(self.dataset[splt])))

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.batch_size,
                          shuffle=self.shuffle_datasets[0], collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.batch_size,
                          shuffle=self.shuffle_datasets[1], collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.batch_size,
                          shuffle=self.shuffle_datasets[2], collate_fn=self.collate_fn,
                          num_workers=self.num_workers)


def collate_batch(batch, feats):
    batch, orig_lengths = pad_batch(batch, feats)
    batch =  {k: torch.stack([b[k] for b in batch])\
              if k in feats or k.endswith('_norm') else [b[k] for b in batch]
              for k in batch[0].keys()}
    batch['orig_lengths'] = orig_lengths
    batch['max_length'] = max(orig_lengths)
    batch['seq_mask'] = LengthMask(orig_lengths)
    batch['seq_pad_mask_adtv'] = LengthMask(orig_lengths).additive_matrix
    batch['seq_pad_mask_bool'] = ~LengthMask(orig_lengths).bool_matrix
    batch['batch_size'] = len(orig_lengths)
    return batch

def pad_batch(batch, feats):
    """
    pad feature tensors to account for different number of frames
    we do NOT zero pad to avoid wierd values in normalisation later in the model
    input:
        - batch: list of input dictionaries
        - feats: list of features to apply padding on (rest left as is)
    returns:
        - padded batch list
        - original length of sequences (could be < n_frames when subsequensing)
    """
    max_frames = max(b['n_frames_orig'].item() for b in batch)
    pad_length = torch.tensor([max_frames - b['n_frames_orig'] for b in batch])
    return (
        [{k: _apply_on_feats(v, k, _pad_n(pad_length[i]), feats)
            for k, v in b.items()}
         for i, b in enumerate(batch)],
        max_frames - pad_length
        )

def _pad_n(n):
    """get padding function for padding x at the first dimension n times"""
    return lambda x: pad(x[None], (0, 0) * (len(x.shape) - 1) + (0, n), "replicate")[0]

def _apply_on_feats(t, name: str, f, feats):
    """apply function f only on features"""
    return f(t) if name in feats or name.endswith('_norm') else t

