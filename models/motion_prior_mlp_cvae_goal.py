"""
Simple baseline for motion generation
"""

from typing import List, Union, Tuple, Optional
import torch
from torch import Tensor
from einops import rearrange, reduce, repeat
from rendering.render_utils import render_motion
from torch.nn.functional import l1_loss, mse_loss
from utils.misc import MLP, freeze
from utils.transformations import (
    transform_body_pose, change_for, remove_z_rot, apply_rot_delta, rot_diff,
    get_z_rot, root2heading
)
import numpy as np
import smplx
from utils.misc import cast_dict_to_numpy

from models.motion_prior_mlp_cvae_goal import MotionPriorMLPCVAEGoal

class Human(MotionPriorMLPCVAEGoal):

    def __init__(self, intention_features, timeless: bool=False,
                 distance_scale: float=2.0, phase_logic=None,
                 max_distance: float=1.0, zero_intention: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.intention_features = ['body_transl_intention_pelv_xy']
        self.intention_goal_feature = 'body_transl_intention_goal'
        self.timeless = timeless
        self.distance_scale = distance_scale
        self.max_distance = max_distance
        self.intention_features = intention_features
        self.phase_logic = phase_logic
        self.zero_intention = zero_intention

    def autoregress(self, init_state_norm: Tensor, init_delta: Tensor,
                    goal: Tuple[Tensor, Tensor], n_steps: int,
                    sample: bool = True, sigma_mult: float = 1.0, latents=None,
                    heading_goal: Optional[Tensor]=None) -> Tensor:
        """
        Starting from an initial x and delta, sample frame by frame for n_steps
        init_state: tensor with features of first state to be fed to the model
        n_steps: how many steps to sample
        sample: whether to sample in the latent space of the VAE
        """
        goal_loc, eta = goal
        states = [init_state_norm]
        if latents is None:
            latents = torch.randn((n_steps, init_state_norm.shape[1], self.latent_dim)).to(self.device) * sigma_mult
        deltas = [init_delta]
        for i in range(n_steps):
            # concatenated features produced by the state and the deltas
            # the difference between x's and states is that x can be more abstract
            # e.g. not including x,y of position of the body etc. The state describes
            # the body motion in an absolute way, the x does not.
            if heading_goal is None:
                intention = self.goal2intention_local(states[-1],
                                                    (goal_loc[i, None], eta[i, None]))
            else:
                intention = self.goal2intention_local(states[-1],
                                                    (goal_loc[i, None], eta[i, None]),
                                                    heading_goal=heading_goal[i, None])
            condition_norm = torch.cat(
                (self.remove_z_from_state(states[-1]), deltas[-1], intention), dim=-1)
            z = latents[i, None]
            delta = self.decode(z, condition_norm)
            deltas.append(delta)
            states.append(self.step(states[-1], delta))
            # add smpl fast forward here
        return {#'mu': torch.cat(mu, dim=0),
                #'logvar': torch.cat(logvar, dim=0),
                'z': latents,
                'delta': torch.cat(deltas[1:], dim=0),
                'generation': torch.cat(states[1:], dim=0)}

    def training_step(self, batch, batch_idx):
        states_norm = self.norm_and_cat(batch, self.state_features)
        deltas_norm = self.norm_and_cat(batch, self.delta_features)
        S = states_norm.shape[0]
        B = states_norm.shape[1]
        goal_idx = batch['intention_goal_frame'].long()
        # when we don't know the goal, we put the same frame as the goal
        # we get which frame is the goal for each frame
        # based on that, we calculate the goal_loc and goal_eta and 
        # call self.goal2intention_local(state, goal)
        J = 22
        # calculate GOAL
        rwrist = rearrange(batch['body_joints'], '... (j d) -> ... j d', j=J)[:, :, 21]
        rwrist_target = torch.gather(rwrist, 1, repeat(goal_idx, 'b s 1 -> b s 3'))
        intention_goal = rearrange(rwrist_target, 'b s d -> s b d')

        # calculate goal orentation from body_orient features
        body_orient_target = torch.gather(batch['body_orient'], 1, repeat(goal_idx, 'b s 1 -> b s 6'))
        body_orient_target = rearrange(body_orient_target, 'b s d -> s b d')
        body_target_heading = root2heading(body_orient_target, in_format="6d")

        # calculate ETA (frames to reach goal) 
        intention_eta = torch.clamp(
            goal_idx - repeat(torch.arange(S), 's -> b s 1', b=B).to(goal_idx.device), min=0)
        intention_eta = rearrange(intention_eta, 'b s ... -> s b ...')

        intention_local = self.goal2intention_local(states_norm,
                                                    goal=(intention_goal, intention_eta),
                                                    heading_goal=body_target_heading)
        # zero out intention when eta is zero
        padding_mask = rearrange(~batch['seq_pad_mask_bool'], 'b s -> s b ()')
        intention_local = intention_local * (intention_eta != 0)
        condition_norm = torch.cat((self.remove_z_from_state(states_norm), deltas_norm,
                                    intention_local), dim=-1)

        ar_steps = self.autoregress_steps
        self.log('autoregress_steps', ar_steps, batch_size=1)
        self.log('kl_beta', self.kl_beta, batch_size=1)

        ## RUN FORWARD PASS ##
        out = self(deltas_norm[1:], condition_norm[:-1], states_norm[:-1])
        delta_norm_gt = deltas_norm[1:]
        ## CALCULATE LOSSES ##
        loss, loss_dict = self.compute_loss_dict(out, delta_norm_gt, padding_mask[:-1])

        # joints loss
        if self.train_hparams.joint_loss:
            joints_gt = rearrange(batch['body_joints'], 'b s (j d) -> s b j d', j=J)
            loss_joints = self.compute_joints_loss(out['generation'], joints_gt[1:], padding_mask[:-1])
            loss = loss + loss_joints
            loss_dict['joints_loss'] = loss_joints

        if self.train_hparams.randomise_ar_steps:
            ar_steps = min(torch.randint(ar_steps + 1, (1,)).item(), condition_norm.shape[0])
            self.log('rnd_autoregress_steps', ar_steps, batch_size=1)
        if ar_steps >= 2:  # use the models predictions autoregressively
            ## RUN AUTOREGRESSIVE PASS ##
            drop_last = torch.rand(1).item() <= 0.5
            (
                states_norm, deltas_norm, padding_mask, joints_gt,
                intention_goal, intention_eta, body_target_heading
            ) = self.chunk_seq_rnd(
                [states_norm, deltas_norm, padding_mask, joints_gt,
                 intention_goal, intention_eta, body_target_heading],
                ar_steps, drop_last)
            out_ar = self.autoregress(states_norm[:1], deltas_norm[:1],
                                      n_steps=states_norm.shape[0] - 1,
                                      goal=(intention_goal, intention_eta),
                                      heading_goal=body_target_heading)
            pred_states_norm = torch.cat((states_norm[:1], out_ar['generation']),
                                         dim=0)
            # corrected_deltas = pred_states - self.unnorm_state(states_norm[:-1])
            # delta_norm_gt = self.norm_delta(corrected_deltas)
            delta_norm_corrected = self.get_state_delta_normed(states1_norm=pred_states_norm[:-1],
                                                               states2_norm=states_norm[1:])
            ## CALCULATE LOSSES ##
            loss_recon_ar, loss_dict_ar = self.compute_reconstruction_loss_dict(
                out_ar, delta_norm_corrected, padding_mask[:-1])
            loss = loss + loss_recon_ar
            loss_dict['recon_loss_ar'] = loss_recon_ar

            if self.train_hparams.joint_loss:
                loss_joints_ar = self.compute_joints_loss(out_ar['generation'], joints_gt[1:], padding_mask[:-1])
                loss = loss + loss_joints_ar
                loss_dict['joints_loss_ar'] = loss_joints_ar

        ## LOG ##
        self.log_dict(loss_dict, on_epoch=True, batch_size=batch['batch_size'], sync_dist=True)

        ## SAVE DATA FOR RENDERING LATER ##
        if batch_idx in self.train_hparams.rendering.train_ids and self.global_rank == 0:
            self.render_data_buffer.append({'batch': batch, 'idx': 0})
        return loss

    def compute_loss_dict(self, out, delta_norm_gt, padding_mask: Tensor = None):
        recon_norm, mu, logvar, delta_norm = (
            out['generation'], out['mu'], out['logvar'], out['delta']
            )

        # KLD loss
        kld = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        if self.train_hparams.kl_free_bit_thrs is not None:
            kld_clamp = torch.clamp(kld, min=self.train_hparams.kl_free_bit_thrs) # free-bit KLD
        kld_loss = (kld_clamp.mean(-1, keepdim=True) * padding_mask).sum() / padding_mask.sum()

        # recostruction loss
        # first frame is always given
        # the goal is to predict the next frame. this is why we don't give
        # the reconstruction for the last frame
        if self.train_hparams.loss_type == "mse":
            recon_loss = mse_loss(delta_norm, delta_norm_gt, reduction='none')
        elif self.train_hparams.loss_type == "l1":
            recon_loss = l1_loss(delta_norm, delta_norm_gt, reduction='none')
        recon_loss_transl = (recon_loss[..., :3].mean(-1, keepdim=True) * padding_mask).sum() / padding_mask.sum()
        recon_loss_rot = (recon_loss[..., 3:9].mean(-1, keepdim=True) * padding_mask).sum() / padding_mask.sum()
        recon_loss_pose = (recon_loss[..., 9:].mean(-1, keepdim=True) * padding_mask).sum() / padding_mask.sum()
        recon_loss = recon_loss_transl + recon_loss_rot + recon_loss_pose
        # recon_loss = (recon_loss.mean(-1, keepdim=True) * padding_mask).sum() / padding_mask.sum()

        loss = recon_loss + self.train_hparams.kl_beta * kld_loss

        # logging KL-div per dimention saves lives
        kld_per_dim = reduce((kld * padding_mask).detach(), 'seq batch dim -> dim', 'sum') /\
            padding_mask.sum()

        loss_dict = {**{
            "kld_loss": kld_loss,
            "recon_loss": recon_loss,
            "recon_loss_ar": torch.zeros_like(recon_loss),
            "joints_loss_ar": torch.zeros_like(recon_loss),
            "recon_loss_transl": recon_loss_transl,
            "recon_loss_rot": recon_loss_rot,
            "recon_loss_pose": recon_loss_pose,
            "loss": loss,
            }, **{f"kld_perdim/dim_{i}": v for i, v in enumerate(kld_per_dim)}}
        return loss, loss_dict

    def validation_step(self, batch, batch_idx):
        states_norm = self.norm_and_cat(batch, self.state_features)
        deltas_norm = self.norm_and_cat(batch, self.delta_features)
        S = states_norm.shape[0]
        B = states_norm.shape[1]
        goal_idx = batch['intention_goal_frame'].long()
        # when we don't know the goal, we put the same frame as the goal
        # we get which frame is the goal for each frame
        # based on that, we calculate the goal_loc and goal_eta and 
        # call self.goal2intention_local(state, goal)
        J = 22
        # calculate GOAL
        rwrist = rearrange(batch['body_joints'], '... (j d) -> ... j d', j=J)[:, :, 21]
        rwrist_target = torch.gather(rwrist, 1, repeat(goal_idx, 'b s 1 -> b s 3'))
        intention_goal = rearrange(rwrist_target, 'b s d -> s b d')

        # calculate goal orentation from body_orient features
        body_orient_target = torch.gather(batch['body_orient'], 1, repeat(goal_idx, 'b s 1 -> b s 6'))
        body_orient_target = rearrange(body_orient_target, 'b s d -> s b d')
        body_target_heading = root2heading(body_orient_target, in_format="6d")

        # calculate ETA (frames to reach goal) 
        intention_eta = torch.clamp(
            goal_idx - repeat(torch.arange(S), 's -> b s 1', b=B).to(goal_idx.device), min=0)
        intention_eta = rearrange(intention_eta, 'b s ... -> s b ...')

        intention_local = self.goal2intention_local(states_norm,
                                                    goal=(intention_goal, intention_eta),
                                                    heading_goal=body_target_heading)
        padding_mask = rearrange(~batch['seq_pad_mask_bool'], 'b s -> s b ()')
        intention_local = intention_local * (intention_eta != 0)
        condition_norm = torch.cat((self.remove_z_from_state(states_norm), deltas_norm,
                                    intention_local), dim=-1)
        # states_norm = self.norm_and_cat(batch, self.state_features)
        # deltas_norm = self.norm_and_cat(batch, self.delta_features)
        # intention_local = self.cat_features(batch, self.intention_features)
        # condition_norm = torch.cat((self.state2x(states_norm), deltas_norm,
        #                             intention_local), dim=-1)

        ## RUN FORWARD PASS ##
        out = self(deltas_norm[1:], condition_norm[:-1], states_norm[:-1])
        delta_norm_gt = deltas_norm[1:]

        ## CALCULATE LOSSES ##
        loss, loss_dict = self.compute_loss_dict(out, delta_norm_gt, padding_mask[:-1])

        self.log_dict(loss_dict, on_step=False,
                 batch_size=batch['batch_size'], sync_dist=True)
        self.log('val/loss_monitor', loss_dict[self.train_hparams.monitor_loss],
                 on_step=False, batch_size=batch['batch_size'], sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        states_norm = self.norm_and_cat(batch, self.state_features)
        deltas_norm = self.norm_and_cat(batch, self.delta_features)
        S = states_norm.shape[0]
        B = states_norm.shape[1]
        goal_idx = batch['intention_goal_frame'].long()
        # when we don't know the goal, we put the same frame as the goal
        # we get which frame is the goal for each frame
        # based on that, we calculate the goal_loc and goal_eta and 
        # call self.goal2intention_local(state, goal)
        J = 22
        # calculate GOAL
        rwrist = rearrange(batch['body_joints'], '... (j d) -> ... j d', j=J)[:, :, 21]
        rwrist_target = torch.gather(rwrist, 1, repeat(goal_idx, 'b s 1 -> b s 3'))
        intention_goal = rearrange(rwrist_target, 'b s d -> s b d')

        # calculate ETA (frames to reach goal) 
        intention_eta = torch.clamp(
            goal_idx - repeat(torch.arange(S), 's -> b s 1', b=B).to(goal_idx.device), min=0)
        intention_eta = rearrange(intention_eta, 'b s ... -> s b ...')

        intention_local = self.goal2intention_local(states_norm,
                                                    (intention_goal, intention_eta))
        condition_norm = torch.cat((self.remove_z_from_state(states_norm), deltas_norm,
                                    intention_local), dim=-1)
        # states_norm = self.norm_and_cat(batch, self.state_features)
        # deltas_norm = self.norm_and_cat(batch, self.delta_features)
        # intention_local = self.cat_features(batch, self.intention_features)
        # condition_norm = torch.cat((self.state2x(states_norm), deltas_norm,
        #                             intention_local), dim=-1)
        padding_mask = rearrange(~batch['seq_pad_mask_bool'], 'b s -> s b ()')

        ## RUN FORWARD PASS ##
        out = self(deltas_norm[1:], condition_norm[:-1], states_norm[:-1])
        delta_norm_gt = deltas_norm[1:]

        ## CALCULATE LOSSES ##
        loss, loss_dict = self.compute_loss_dict(out, delta_norm_gt, padding_mask[:-1])

        self.log('test/losses', loss_dict, on_step=False,
                 batch_size=batch['batch_size'])

        ## SAVE DATA FOR RENDERING LATER ##
        if batch_idx in self.train_hparams.rendering.test_ids and self.global_rank == 0:
            self.render_data_buffer.append({'batch': batch, 'idx': 0})
        return loss

    @torch.no_grad()
    def render_buffer(self, buffer: list[dict], epoch_in_fname=False):
        """
        Get a list of dicts in the form of {'batch':..., 'idx': n}
        """
        video_names = []
        # create videos and save full paths
        for data in buffer:

            # RUN FWD PASS
            batch = data['batch']
            batch_idx = data['idx']
            states_norm = self.norm_and_cat(batch, self.state_features)
            deltas_norm = self.norm_and_cat(batch, self.delta_features)
            goal = rearrange(batch['goal'],
                                        'b s d -> s b d')
            rm_pad = lambda x, idx: x[:batch['orig_lengths'][idx],idx, None]

            # remove zero-padded frames as well as the last frame
            # and take first sample of the batch
            # condition_norm = condition_norm[:batch['orig_lengths'][batch_idx],batch_idx, None]
            states_norm = rm_pad(states_norm, batch_idx)
            deltas_norm = rm_pad(deltas_norm, batch_idx)
            goal = rm_pad(goal, batch_idx)

            # AUTOREGRESS
            n_steps = self.train_hparams.rendering.ar_steps
            goal_loc = repeat(goal[0], 'b d -> s b d', s=n_steps+1)
            goal_eta = repeat(torch.flip(torch.arange(goal_loc.shape[0]), dims=[0]),
                              's -> s b 1', b=goal_loc.shape[1]).to(goal_loc.device)
            out = self.autoregress(states_norm[:1], deltas_norm[:1],
                                   n_steps=n_steps, goal=(goal_loc[:-1], goal_eta[:-1]))
            states_norm = torch.cat((states_norm[:1], out['generation']), dim=0)
            unnorm_states = self.uncat_inputs(self.unnorm_state(states_norm),
                                              self.state_dims)
            datum = {name: feats for name, feats in zip(self.state_features,
                                                        unnorm_states)}
            intentions = self.goal2intention(states_norm, (goal_loc, goal_eta))
            datum = datum | intentions
            pred_joints = self.forward_smpl(states_norm, fast=True)
            datum['body_joints'] = pred_joints.detach().cpu()
            datum['goal'] = goal_loc.detach().cpu()
            filename = self.get_video_filename(batch['filename'][batch_idx],
                                               epoch_in_fname=epoch_in_fname,
                                               tags=['_autoregress'])
            # RENDER THE MOTION
            render_motion(self.renderer,cast_dict_to_numpy(datum), filename, pose_repr="6d")
            video_names.append(filename)

        return video_names

    def rollout(self, init_state: dict, n_steps: int, goal: Tuple[Tensor, Tensor]=None,
                sample: bool = True, sigma_mult: float = 1.0, latents=None,
                fwd_smpl=False, fast=True, return_vertices=False):
        """
        starting from an initial state, sample frame by frame for n_steps
        init_state: dict with raw features of pose, transl, orient at
            the initial state 
        n_steps: how many steps to sample
        sample: whether to sample in the latent space of the VAE
        """
        if goal is None:
            goal = rearrange(init_state['goal'], 'b s d -> s b d')
            goal_loc = repeat(goal[0], 'b d -> s b d', s=n_steps+1)
            goal_eta = repeat(torch.flip(torch.arange(goal_loc.shape[0]), dims=[0]),
                                's -> s b 1', b=goal_loc.shape[1]).to(goal_loc.device)
        else:
            goal_loc, goal_eta = goal
        states_norm = self.norm_and_cat(init_state, self.state_features)[25:26]
        deltas_norm = self.norm_and_cat(init_state, self.delta_features)[25:26]
        # deltas_norm = torch.zeros_like(states_norm)
        out = self.autoregress(states_norm[:1], deltas_norm[:1], n_steps=n_steps,
                               goal=(goal_loc, goal_eta), sigma_mult=sigma_mult,
                               sample=sample, latents=latents)
        pred_states_norm = torch.cat((states_norm[:1], out['generation']), dim=0)
        pred_states = self.unnorm_state(pred_states_norm)
        states_list = self.uncat_inputs(pred_states, self.state_dims)
        pred_smpl_params = {name: feats for name, feats in zip(self.state_features, states_list)}
        intentions = self.goal2intention(pred_states_norm, (goal_loc, goal_eta))
        pred_smpl_params = pred_smpl_params | intentions
        pred_smpl_params['goal'] = goal_loc.detach().cpu()
        if fwd_smpl:
            S, B = pred_smpl_params['body_transl'].shape[:2]
            flat_aa = lambda x: transform_body_pose(
                rearrange(x, 's b ... -> (s b) ...'), '6d->aa')
            body_transl = rearrange(pred_smpl_params['body_transl'],
                                    's b ... -> (s b) ...')
            body_pose = flat_aa(pred_smpl_params['body_pose'])
            body_orient = flat_aa(pred_smpl_params['body_orient'])
            smpl_output = self.run_smpl_fwd(
                body_transl=body_transl,
                body_pose=body_pose,
                body_orient=body_orient,
                fast=fast)
            pred_joints = smpl_output.joints
            pred_joints = rearrange(pred_joints, '(s b) ... -> s b ...', s=S, b=B)
            if not return_vertices:
                return pred_smpl_params, out, pred_joints
            else:
                pred_vertices = smpl_output.vertices
                pred_vertices = rearrange(pred_vertices, '(s b) ... -> s b ...', s=S, b=B)
                return pred_smpl_params, out, pred_joints, pred_vertices

            
        else:
            return pred_smpl_params, out, None



    def goal2intention(self, state_norm: Tensor, goal: Tuple[Tensor, Union[Tensor, int]],
                             heading_goal: Tensor=None):
        """
        For visualisation purposes. Not used by the model.
        """
        goal_loc, goal_eta = goal
        if self.timeless:
            goal_eta = None
        intentions = {}
        state = self.unnorm_state(state_norm)
        joints = self.forward_smpl(state_norm, fast=True)

        rwrist = joints[:, :, 21]
        pelvis = joints[:, :, 0]
        R_z = get_z_rot(state[..., 3:9], in_format='6d')

        phase_in = 1
        phase_out = 1
        if self.phase_logic is not None:
            # distance to goal
            dtg = torch.norm((goal_loc - pelvis)[..., :2], dim=-1, keepdim=True)
            alpha = self.phase_logic.phase_scale
            offset = self.phase_logic.phase_distance
            phase_in = torch.exp(alpha*(-dtg + offset))/\
                (1 + torch.exp(alpha*(-dtg + offset)))
            phase_out = 1 - phase_in

        # Calculate needed intention features
        if "right_wrist_intention_pelv_xy_exp_z" in self.intention_features:
            rwrist_intention_pelv_xy = joint_intention_pelv_xy_exp_z(
                rwrist, goal_loc, state[..., 3:9], goal_eta,
                z_scale=self.distance_scale, in_format='6d')
            # convert to global
            rwrist_intention_pelv_xy = change_for(rwrist_intention_pelv_xy,
                                                  R_z,
                                                  forward=False)
            intentions['right_wrist_intention_pelv_xy_exp_z'] = rwrist_intention_pelv_xy * phase_in

        if "right_wrist_intention_pelv_xy" in self.intention_features:
            rwrist_intention_pelv_xy = joint_intention_pelv_xy(
                rwrist, goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance)
            rwrist_intention_pelv_xy = change_for(rwrist_intention_pelv_xy,
                                                  R_z,
                                                  forward=False)
            intentions['right_wrist_intention_pelv_xy'] = rwrist_intention_pelv_xy * phase_in

        if "body_transl_intention_pelv_xy" in self.intention_features:
            transl_intention_pelv_xy = joint_intention_pelv_xy(
                state[..., :3], goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance)
            transl_intention_pelv_xy[..., 2] = 0
            transl_intention_pelv_xy = change_for(transl_intention_pelv_xy,
                                                  R_z,
                                                  forward=False)
            intentions['body_transl_intention_pelv_xy'] = transl_intention_pelv_xy * phase_out

        if "pelvis_transl_exp_delta_pelv_xy" in self.intention_features:
            transl_intention_pelv_xy = joint_intention_pelv_xy(
                pelvis, goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance, force_timeless=True)
            transl_intention_pelv_xy[..., 2] = 0
            transl_intention_pelv_xy = change_for(transl_intention_pelv_xy,
                                                  R_z,
                                                  forward=False)
            intentions['pelvis_transl_exp_delta_pelv_xy'] = transl_intention_pelv_xy * phase_out

        if "pelvis_transl_intention_pelv_xy" in self.intention_features:
            transl_intention_pelv_xy = joint_intention_pelv_xy(
                pelvis, goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance)
            transl_intention_pelv_xy[..., 2] = 0
            transl_intention_pelv_xy = change_for(transl_intention_pelv_xy,
                                                  R_z,
                                                  forward=False)
            intentions['pelvis_transl_intention_pelv_xy'] = transl_intention_pelv_xy * phase_out

        if "body_z_orient_exp_delta_intention_pelv_xy" in self.intention_features:
            if heading_goal is None:  # use goal_loc as heading goal
                heading_goal = goal_loc - pelvis
                heading_goal[..., 2] = 0
            heading_intention_local = body_z_orient_intention_pelv_xy(
                state[..., 3:9], heading_goal, goal_eta, in_format='6d', force_timeless=True)
            heading_intention_local = change_for(heading_intention_local,
                                                  R_z,
                                                  forward=False)
            intentions['body_z_orient_exp_delta_intention_pelv_xy'] = heading_intention_local * phase_out

        if "body_z_orient_intention_pelv_xy" in self.intention_features:
            if heading_goal is None:  # use goal_loc as heading goal
                heading_goal = goal_loc - pelvis
                heading_goal[..., 2] = 0
            heading_intention_local = body_z_orient_intention_pelv_xy(
                state[..., 3:9], heading_goal, goal_eta, in_format='6d')
            heading_intention_local = change_for(heading_intention_local,
                                                  R_z,
                                                  forward=False)
            intentions['body_z_orient_intention_pelv_xy'] = heading_intention_local * phase_out

        if "right_wrist_exp_distance" in self.intention_features:
            exp_dist = joint_exp_distance(rwrist, goal_loc,
                                          scale=self.distance_scale, max_value=self.max_distance)
            intentions['right_wrist_exp_distance'] = exp_dist * phase_in

        if "pelvis_xy_exp_distance" in self.intention_features:
            exp_dist = joint_exp_distance(pelvis[..., :2], goal_loc[..., :2],
                                          scale=self.distance_scale, max_value=self.max_distance)
            intentions['pelvis_xy_exp_distance'] = exp_dist * phase_out

        if self.zero_intention:
            intentions = {k: torch.zeros_like(v) for k, v in intentions.items()}
        return intentions
        # return {k: torch.clamp(v, min=-2, max=2) for k, v in intentions.items()}

    def goal2intention_local(self, state_norm: Tensor, goal: Tuple[Tensor, Union[Tensor, int]],
                             heading_goal: Tensor=None):
        """
        What do I have to do in order to achieve my goal?
        """
        goal_loc, goal_eta = goal
        if self.timeless:
            goal_eta = None
        intentions = {}
        state = self.unnorm_state(state_norm)
        joints = self.forward_smpl(state_norm, fast=True)

        rwrist = joints[:, :, 21]
        pelvis = joints[:, :, 0]

        phase_in = 1
        phase_out = 1
        if self.phase_logic is not None:
            # distance to goal
            dtg = torch.norm((goal_loc - pelvis)[..., :2], dim=-1, keepdim=True)
            alpha = self.phase_logic.phase_scale
            offset = self.phase_logic.phase_distance
            phase_in = torch.exp(alpha*(-dtg + offset))/\
                (1 + torch.exp(alpha*(-dtg + offset)))
            phase_out = 1 - phase_in

        # Calculate needed intention features
        if "right_wrist_intention_pelv_xy_exp_z" in self.intention_features:
            rwrist_intention_pelv_xy = joint_intention_pelv_xy_exp_z(
                rwrist, goal_loc, state[..., 3:9], goal_eta,
                z_scale=self.distance_scale, in_format='6d')
            intentions['right_wrist_intention_pelv_xy_exp_z'] = rwrist_intention_pelv_xy * phase_in

        if "right_wrist_intention_pelv_xy" in self.intention_features:
            rwrist_intention_pelv_xy = joint_intention_pelv_xy(
                rwrist, goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance)
            intentions['right_wrist_intention_pelv_xy'] = rwrist_intention_pelv_xy * phase_in

        if "body_transl_intention_pelv_xy" in self.intention_features:
            transl_intention_pelv_xy = joint_intention_pelv_xy(
                state[..., :3], goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance)
            transl_intention_pelv_xy[..., 2] = 0
            intentions['body_transl_intention_pelv_xy'] = transl_intention_pelv_xy * phase_out

        if "pelvis_transl_intention_pelv_xy" in self.intention_features:
            transl_intention_pelv_xy = joint_intention_pelv_xy(
                pelvis, goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance)
            transl_intention_pelv_xy[..., 2] = 0
            intentions['pelvis_transl_intention_pelv_xy'] = transl_intention_pelv_xy * phase_out

        if "pelvis_transl_exp_delta_pelv_xy" in self.intention_features:
            transl_intention_pelv_xy = joint_intention_pelv_xy(
                pelvis, goal_loc, state[..., 3:9], goal_eta, in_format='6d',
                scale=self.distance_scale, max_value=self.max_distance, force_timeless=True)
            transl_intention_pelv_xy[..., 2] = 0
            intentions['pelvis_transl_exp_delta_pelv_xy'] = transl_intention_pelv_xy * phase_out

        if "body_z_orient_exp_delta_intention_pelv_xy" in self.intention_features:
            if heading_goal is None:  # use goal_loc as heading goal
                heading_goal = goal_loc - pelvis
                heading_goal[..., 2] = 0
            heading_intention_local = body_z_orient_intention_pelv_xy(
                state[..., 3:9], heading_goal, goal_eta, in_format='6d', force_timeless=True)
            intentions['body_z_orient_exp_delta_intention_pelv_xy'] = heading_intention_local * phase_out

        if "body_z_orient_intention_pelv_xy" in self.intention_features:
            if heading_goal is None:  # use goal_loc as heading goal
                heading_goal = goal_loc - pelvis
                heading_goal[..., 2] = 0
            heading_intention_local = body_z_orient_intention_pelv_xy(
                state[..., 3:9], heading_goal, goal_eta, in_format='6d')
            intentions['body_z_orient_intention_pelv_xy'] = heading_intention_local * phase_out

        if "right_wrist_exp_distance" in self.intention_features:
            exp_dist = joint_exp_distance(rwrist, goal_loc,
                                          scale=self.distance_scale, max_value=self.max_distance)
            intentions['right_wrist_exp_distance'] = exp_dist * phase_in

        if "pelvis_xy_exp_distance" in self.intention_features:
            exp_dist = joint_exp_distance(pelvis[..., :2], goal_loc[..., :2],
                                          scale=self.distance_scale, max_value=self.max_distance)
            intentions['pelvis_xy_exp_distance'] = exp_dist * phase_out

        if "rwrist_full_local_intention" in self.intention_features:
            heading_goal = goal_loc - pelvis
            heading_goal[..., 2] = 0
            intention = full_local_intention(pelvis_orient=state[..., 3:9],
                                             joint=rwrist,
                                             joint_goal=goal_loc,
                                             heading_goal=heading_goal,
                                             k=goal_eta,
                                             in_format='6d')
            intentions['rwrist_full_local_intention'] = intention

        if self.zero_intention:
            intentions = {k: torch.zeros_like(v) for k, v in intentions.items()}
        return torch.cat([intentions[f] for f in self.intention_features], dim=-1)
        # return torch.cat([torch.clamp(intentions[f],min=-2, max=2) for f in self.intention_features], dim=-1)
