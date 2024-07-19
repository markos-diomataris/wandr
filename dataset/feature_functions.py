from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from utils.transformations import change_for, get_z_rot, root2heading
from torch.nn.functional import normalize


def joint_intention_pelv_xy_exp_z(joint: Tensor, joint_goal:Tensor,
                                  pelvis_orient: Tensor, eta: Optional[int]=None,
                                  z_scale: float=1.0, in_format: str="aa",
                                  **kwargs):
    joint_int = joint_goal - joint
    R_z = get_z_rot(pelvis_orient, in_format=in_format)
    # rotate -R_z
    joint_int_pelv = change_for(joint_int, R_z, forward=True)
    # calculate pelvis to goal euclidean distance
    distance_to_goal = torch.norm(joint_int, dim=-1)
    exp_mult = torch.exp(-z_scale*distance_to_goal)
    joint_int_pelv[..., 2] *= exp_mult
    if eta is not None:
        # if we have a time constraint, we need to scale the intention
        Dt = eta/30 + 1e-5
        return joint_int_pelv / Dt
    else:
        return  exp_scale_norm(joint_int_pelv, **kwargs)

def joint_intention_pelv_xy(joint: Tensor, joint_goal:Tensor,
                            pelvis_orient: Tensor, eta: Optional[int]=None,
                            in_format: str="aa", scale: float=1.0, force_timeless: bool=False,
                            **kwargs):
    joint_int = joint_goal - joint
    R_z = get_z_rot(pelvis_orient, in_format=in_format)
    # rotate -R_z
    joint_int_pelv = change_for(joint_int, R_z, forward=True)
    if eta is not None and not force_timeless:
        # if we have a time constraint, we need to scale the intention
        Dt = eta/30 + 1e-5
        return joint_int_pelv / Dt
    else:
        return  exp_scale_norm(joint_int_pelv, **kwargs)

def body_z_orient_intention_pelv_xy(pelvis_orient: Tensor, heading_goal: Tensor,
                                    eta: Optional[int]=None, in_format: str="aa",
                                    force_timeless: bool=False):
    """
    pelvis_orient: body orientation 
    heading_goal: a vector that indicates the desired heading
    eta: how many frames remain towards achieving it
    returns the (heading_goal-current_heading)/time rotated by the pelvis z orientation
    this way this vector becomes invariant to the body's global z orientation
    """
    # make sure heading is a unit vector
    heading_goal = normalize(heading_goal, dim=-1)
    heading_vec = root2heading(pelvis_orient, in_format=in_format)
    heading_int =  heading_goal - heading_vec
    R_z = get_z_rot(pelvis_orient, in_format=in_format)
    # rotate -R_z
    heading_int_pelv = change_for(heading_int, R_z, forward=True)
    if eta is not None and not force_timeless:
        # if we have a time constraint, we need to scale the intention
        Dt = eta/30 + 1e-5
        return heading_int_pelv / Dt
    else:
        return heading_int_pelv

def full_local_intention(pelvis_orient: Tensor, joint: Tensor,
                         joint_goal: Tensor, heading_goal: Tensor,
                         eta: Optional[int]=None, in_format: str="aa",
                         **kwargs):
    assert False, 'this function is no longer supported'
    # intention for right wrist
    joint_intention = joint_intention_pelv_xy(joint, joint_goal, pelvis_orient, eta, in_format)
    heading_intention = body_z_orient_intention_pelv_xy(pelvis_orient, heading_goal, eta, in_format)
    intention_exp_dist = joint_exp_distance(joint, joint_goal)
    return torch.cat([joint_intention, heading_intention, intention_exp_dist], dim=-1)
    
def exp_scale_norm(x: Tensor, scale: float=1.0, max_value: float=1.0):
    """
    scale the norm of x by -exp(-norm(x)*scale) + 1
    """
    x_norm = -torch.exp(
        -torch.linalg.norm(x, dim=-1, keepdim=True)) + 1
    return normalize(x, dim=-1) * x_norm * max_value

def joint_exp_distance(joint: Tensor, joint_goal:Tensor, scale: float=1, max_value: float=1.0):
    joint_int = joint_goal - joint
    return (-torch.exp(-joint_int.norm(dim=-1, keepdim=True) * scale) + 1) * max_value