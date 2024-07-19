"""
Simple baseline for motion generation
"""

from os.path import exists

from typing import List
import torch
from torch import Tensor
import numpy as np
from einops import rearrange
from utils.misc import cast_dict_to_tensors
from utils.transformations import (
    rot_diff, transform_body_pose, change_for, get_z_rot, remove_z_rot, apply_rot_delta
)
from utils.misc import freeze
import smplx
from utils.smpl_fast import smplx_forward_fast
from pytorch_lightning import LightningModule

class MotionModel():
    """
    This class keeps all the motion related functions and models (smplx)
    It has nothing to do with training neural networks
    The two main functionalities are:
    - normalisation of motion data
    - forward kinematics using smplx
    """
    def __init__(self, 
                 statistics_path: str, smplx_path: str, smpl_type: str,
                 norm_type: str,
                 feats_dims,  state_features: List[str],
                 delta_features: List[str],
                 device,
                 **kwargs):
        super().__init__()

        self.smplx_path = smplx_path
        self.norm_type = norm_type
        self.smpl_type = smpl_type
        
        # feature related variables
        self.feats_dims = feats_dims
        self.state_features = state_features
        self.delta_features = delta_features
        self.state_dims = [self.feats_dims[feat] for feat in state_features]
        
        # Load normalisation statistics
        self.stats = self.load_norm_statistics(statistics_path, device)

        # add fast smpl forward function and instantiate smplx model
        setattr(smplx.SMPLX, 'smplx_forward_fast', smplx_forward_fast)
        self.body_model = smplx.create(smplx_path.removeprefix('/lustre'),
                                model_type=self.smpl_type,
                                gender='neutral',
                                ).to(self.device).eval();
        freeze(self.body_model)
        
    def run_smpl_fwd(self, body_transl, body_orient, body_pose, fast=True):
        batch_size = body_transl.shape[0]
        self.body_model.batch_size = batch_size
        fwd_fn = self.body_model.smplx_forward_fast if fast \
                 else self.body_model.forward
        return fwd_fn(transl=body_transl,
                               body_pose=body_pose,
                               global_orient=body_orient,
                               # just zero-out everything else
                               expression=torch.zeros((batch_size, self.body_model.expression.shape[-1])).to(body_transl.device),
                               left_hand_pose=torch.zeros((batch_size, self.body_model.left_hand_pose.shape[-1])).to(body_transl.device),
                               right_hand_pose=torch.zeros((batch_size, self.body_model.right_hand_pose.shape[-1])).to(body_transl.device),
                               jaw_pose=torch.zeros((batch_size, 3)).to(body_transl.device),
                               reye_pose=torch.zeros((batch_size, 3)).to(body_transl.device),
                               leye_pose=torch.zeros((batch_size, 3)).to(body_transl.device))

    def forward_smpl(self, states, fast=True, states_are_normed=True):
        """
        Get a normalised state conaining [trans,orient,pose] and return
        global joints by running smpl[x] forward
        """
        if states_are_normed:
            states = self.unnorm_state(states)

        unnorm_states = self.uncat_inputs(states, self.state_dims)
        pred_smpl_params = {
            name: feats for name, feats in zip(self.state_features, unnorm_states)}
        S, B = pred_smpl_params['body_transl'].shape[:2]
        flat_aa = lambda x: transform_body_pose(
            rearrange(x, 's b ... -> (s b) ...'), '6d->aa')
        body_transl = rearrange(pred_smpl_params['body_transl'],
                                's b ... -> (s b) ...')
        body_pose = flat_aa(pred_smpl_params['body_pose'])
        body_orient = flat_aa(pred_smpl_params['body_orient'])
        joints = self.run_smpl_fwd(
            body_transl=body_transl,
            body_pose=body_pose,
            body_orient=body_orient,
            fast=fast).joints
        joints = rearrange(joints, '(s b) ... -> s b ...', s=S, b=B)
        return joints

    def step(self, state_norm, delta_norm):
        """"
        Given a state [translation, orientation, pose] and state deltas,
        properly calculate the next state
        input and output are normalised features hence we first unnormalise,
        perform the calculations and then normalise again
        """
        # check set equality
        if self.delta_features == ['body_transl', 'body_orient', 'body_pose']:
            # predicted deltas are actual states => no need to do anything
            return delta_norm
        

        # unnorm features
        state = self.unnorm_state(state_norm)
        delta = self.unnorm_delta(delta_norm)

        # apply deltas
        # get velocity in global c.f. and add it to the state position

        if 'body_transl_delta_pelv_xy' in self.delta_features:
            pelvis_orient = state[..., 3:9]
            R_z = get_z_rot(pelvis_orient, in_format="6d")
            # rotate R_z
            root_vel =  change_for(delta[..., :3], R_z, forward=False)
        elif 'body_transl_delta_pelv' in self.delta_features:
            pelv_orient = transform_body_pose(state[..., 3:9], "6d->rot")
            root_vel =  change_for(delta[..., :3], pelv_orient, forward=False)
        new_state_pos = state[..., :3] + root_vel

        # apply rotational deltas
        new_state_rot = apply_rot_delta(state[..., 3:], delta[..., 3:],
                                        in_format="6d", out_format="6d")

        # cat and normalise the result
        new_state = torch.cat((new_state_pos, new_state_rot), dim=-1)
        new_state_norm = self.norm_state(new_state)
        return new_state_norm

    # IO FUNCTIONS

    def get_state_delta_normed(self, states1_norm: Tensor, states2_norm: Tensor) -> Tensor:
        # unnormalise states
        states1 = self.unnorm_state(states1_norm)
        states2 = self.unnorm_state(states2_norm)

        # get pelvis global velocity and transform to local
        vel_glob = states2[..., :3] - states1[..., :3]
        if 'body_transl_delta_pelv' in self.delta_features:
            pelvis_orient = transform_body_pose(states1[..., 3:9], "6d->rot")
            root_vel_local =  change_for(vel_glob, pelvis_orient)
        elif 'body_transl_delta_pelv_xy' in self.delta_features:
            pelvis_orient = states1[..., 3:9]
            R_z = get_z_rot(pelvis_orient, in_format="6d")
            root_vel_local =  change_for(vel_glob, R_z, forward=True)

        # get rotational differences between states for rotation features
        rot_diffs = rot_diff(rots1=states1[..., 3:], rots2=states2[..., 3:],
                            in_format="6d", out_format="6d")
        
        delta = torch.cat((root_vel_local, rot_diffs), dim=-1)
        delta_norm = self.norm_delta(delta)
        return delta_norm

    def norm_and_cat(self, batch, features):
        """
        turn batch data into the format the forward() function expects
        """
        ## PREPARE INPUT ##
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        # get list of all features that we will give as an input to the VAE
        x_list = [seq_first(batch[name]) for name in features]
        # normalise and cat to a unified feature vector
        x_list_norm = self.norm_inputs(x_list, features)
        x_norm, _ = self.cat_inputs(x_list_norm)
        return x_norm

    def cat_inputs(self, x_list: List[Tensor]):
        """
        cat the inputs to a unified vector and return their lengths in order
        to un-cat them later
        """
        return torch.cat(x_list, dim=-1), [x.shape[-1] for x in x_list]
    
    def uncat_inputs(self, x: Tensor, lengths: List[int]):
        """
        split the unified feature vector back to its original parts
        """
        return torch.split(x, lengths, dim=-1)

    def remove_z_from_state(self, state_norm):
        # drop global information that relates to Z-rotational symmetry
        # i.e. Z euler angle for root orientation and x,y positions for
        # pelvis translation
        state = self.unnorm_state(state_norm)
        body_orient_xy = remove_z_rot(state[..., 3:9], in_format="6d")
        body_orient_xy_norm = self.norm(body_orient_xy, self.stats['body_orient_xy'])
        pos_z_norm = state_norm[..., 2:3]
        return torch.cat((pos_z_norm, body_orient_xy_norm, state_norm[..., 9:]),
                         dim=-1) 

    @staticmethod
    def load_norm_statistics(path: str, device):
        path = path.removeprefix('/lustre')
        assert exists(path), f"{path} stats file does not exist"
        stats = np.load(path, allow_pickle=True)[()]
        return cast_dict_to_tensors(stats, device=device)

    def batch2input(self, batch):
        """
        turn batch data into the format the forward() function expects
        """
        ## PREPARE INPUT ##
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        # get list of all features that we will give as an input to the VAE
        x_list = [seq_first(batch[name]) for name in self.state_features]
        # normalise and cat to a unified feature vector
        x_list_norm = self.norm_inputs(x_list, self.state_features)
        x_norm, lengths = self.cat_inputs(x_list_norm)
        return x_norm, lengths
    
    def rollout_sanity_check(self, batch):
        """
        starting from an initial state, sample frame by frame for n_steps
        init_state: dict with raw features of pose, transl, orient at
            the initial state 
        n_steps: how many steps to sample
        sample: whether to sample in the latent space of the VAE
        """
        states_norm = self.norm_and_cat(batch, self.state_features)
        # deltas_norm = self.get_state_delta_normed(states_norm[:-1], states_norm[1:])
        deltas_norm = self.norm_and_cat(batch, self.delta_features)[1:]
        states_list = [states_norm[:1]]
        for d in deltas_norm:
            states_list.append(self.step(states_list[-1], d[None]))
        pred_states_norm = torch.cat((states_list), dim=0)
        pred_states = self.unnorm_state(pred_states_norm)
        states_list = self.uncat_inputs(pred_states, self.state_dims)
        return {name: feats for name, feats in zip(self.state_features, states_list)}

    @staticmethod
    def chunk_seq_rnd(x_list: list[Tensor], chunk_size: int, drop_last: bool):
        """
        Split sequences to sub-sequences and push the extra dimention to the batch
        """
        drop = x_list[0].shape[0] % chunk_size
        if drop == 0:
            chunk_seq_fn = lambda x : rearrange(x, '(j k) b ... -> k (b j) ...',
                                                k=chunk_size)
        else:
            if drop_last:
                chunk_seq_fn = lambda x : rearrange(x[:-drop],
                                                    '(j k) b ... -> k (b j) ...', k=chunk_size) \
                                                        if x is not None else None
            else:
                chunk_seq_fn = lambda x : rearrange(x[drop:],
                                                    '(j k) b ... -> k (b j) ...', k=chunk_size) \
                                                        if x is not None else None
        return [chunk_seq_fn(x) for x in x_list]

    # NORMALISATION FUNCTIONS

    def norm(self, x, stats):
        if self.norm_type == "std":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return (x - mean) / (std + 1e-5)
        elif self.norm_type == "norm":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            assert ((x - min) / (max - min + 1e-5)).min() >= 0
            assert ((x - min) / (max - min + 1e-5)).max() <= 1
            return (x - min) / (max - min + 1e-5)

    def unnorm(self, x, stats):
        if self.norm_type == "std":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return x * (std + 1e-5) + mean
        elif self.norm_type == "norm":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            return x * (max - min + 1e-5) + min

    def unnorm_state(self, state_norm: Tensor) -> Tensor:
        # unnorm state
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(state_norm, self.state_dims),
                               self.state_features))[0]
        
    def unnorm_delta(self, delta_norm: Tensor) -> Tensor:
        # unnorm delta
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(delta_norm, self.delta_dims),
                               self.delta_features))[0]

    def norm_state(self, state:Tensor) -> Tensor:
        # normalise state
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(state, self.state_dims),
                             self.state_features))[0]

    def norm_delta(self, delta:Tensor) -> Tensor:
        # normalise delta
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(delta, self.delta_dims),
                             self.delta_features))[0]

    def norm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Normalise inputs using the self.stats metrics
        """
        x_norm = []
        for x, name in zip(x_list, names):
            x_norm.append(self.norm(x, self.stats[name]))
        return x_norm

    def unnorm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Un-normalise inputs using the self.stats metrics
        """
        x_unnorm = []
        for x, name in zip(x_list, names):
            x_unnorm.append(self.unnorm(x, self.stats[name]))
        return x_unnorm

    def view_point_from_pelvis(self, state_norm: Tensor, point: Tensor):
        state = self.unnorm_state(state_norm)
        body_transl = state[..., :3]
        R_z = get_z_rot(state[..., 3:9], in_format="6d")
        return change_for(point, R_z, T=body_transl, forward=True)
