from copy import copy
import shortuuid
import numpy as np
import torch
from torch import Tensor
import logging
import os
import random
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from hydra.utils import get_class

# A logger for this file
log = logging.getLogger(__name__)

def to_tensor(array):
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array)

def parse_npz(filename, allow_pickle=True, framerate_ratio=None,
              chunk_start=None, chunk_duration=None, load_joints=True,
              undo_interaction=False, trim_nointeractions=False):
    npz = np.load(filename, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    npz['chunk_start'] = 0

    if load_joints:
        # get precomputed joint features if they exist
        l = filename.split('/')
        l.insert(-2, 'joints')
        l = os.path.join('/',*l)
        npz_joints = l[:-4] + '_joints' + l[-4:]
        npz_joints = np.load(npz_joints, allow_pickle=allow_pickle)
        npz_joints = {k: npz_joints[k] for k in npz_joints.files}
        npz = {**npz, **npz_joints}

    if framerate_ratio is not None:
        # reduce framerate
        assert isinstance(framerate_ratio, int)
        old_framerate = npz['framerate']
        new_framerate = old_framerate / framerate_ratio
        npz = subsample(npz, framerate_ratio)
        npz['framerate'] = new_framerate
        npz['n_frames'] = npz['body']['params']['transl'].shape[0]

    # undo interaction
    if undo_interaction:
        # TODO: also zero out contact on human
        # no contact to the object
        npz['contact']['object'] = np.zeros_like(npz['contact']['object'])
        # no motion to the object
        temp = np.zeros_like(npz['object']['params']['transl'])
        temp += npz['object']['params']['transl'][:1]
        npz['object']['params']['transl'] = temp
        # no rotation to the object
        npz['object']['params']['global_orient'] = np.zeros_like(npz['object']['params']['global_orient'])

    if trim_nointeractions:
        # trim start and end of clip where no interactions take place
        contact = npz['contact']['object']
        contact_frames = np.nonzero(contact)[0]
        contact_start = contact_frames.min()
        contact_length = contact_frames.max() - contact_start
        npz = cut_chunk(npz, contact_start, contact_length)

    # cut to smaller continuous chunks
    if chunk_duration is not None:
        chunk_length = min(int(chunk_duration * npz['framerate']), npz['n_frames'])
        if chunk_start is None:
            chunk_start = random.randint(0, npz['n_frames'] - chunk_length)
        npz = cut_chunk(npz, chunk_start, chunk_length)
        
    return DotDict(npz)

def cut_chunk(npz, chunk_start, chunk_length):
    """
    cut a chunk of a sequence of length chunk_length | let's get functional here :P
    """
    npz = _cut_chunk(npz, chunk_start=chunk_start, chunk_length=chunk_length)
    # readjust metadata
    if 'trans' in npz.keys():
        npz['n_frames'] = npz['trans'].shape[0]
    else:
        npz['n_frames'] = npz['body']['params']['transl'].shape[0]
    npz['chunk_start'] = chunk_start
    if 'goal_frame' in npz.keys():
        if npz['goal_frame'] >= (chunk_start + chunk_length + 1) or npz['goal_frame'] <= chunk_start:
            # NOTE: the +1 should not be there, its a data problem whre the goal is on the last frame
            # but the idx is +1 bigger
            # if goal is beyond the chunk, then remove it completely
            del npz['goal_frame']
        else:
            npz['goal_frame'] -= chunk_start
    return npz

def _cut_chunk(npz, chunk_start, chunk_length):
    if isinstance(npz, np.ndarray) or isinstance(npz, Tensor):
        return npz[chunk_start:chunk_start + chunk_length]
    elif isinstance(npz, dict):
        return {k: _cut_chunk(v, chunk_start, chunk_length)
                for k, v in npz.items()}
    else:
        return npz

def subsample(npz, ratio):
    """
    Subsample 0-dim (frames) | let's get functional here :P
    """
    if isinstance(npz, np.ndarray) or isinstance(npz, Tensor):
        return npz[::ratio]
    elif isinstance(npz, dict):
        return {k: subsample(v, ratio) for k, v in npz.items()}
    else:
        return npz

# def prepare_params(params, frame_mask, dtype = np.float32):
#     return {k: v[frame_mask].astype(dtype) for k, v in params.items()}

def DotDict(in_dict):
    if isinstance(in_dict, dotdict):
        return in_dict 
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
        if isinstance(v,dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def sequential(dims, layernorm=False, end_with=None, nonlinearity=torch.nn.ReLU,
               dropout=0.0):
    """
    Instantiate a Sequential with ReLUs
    dims: list with integers specifying the dimentionality of the mlp
    e.g. if you want three layers dims should look like [d1, d2, d3, d4]. d1 is
    the input and d4 the output dimention.
    if dims == [] or dims == [k] then you get the identity module
    """
    if len(dims) <= 1:
        return torch.nn.Identity()

    def linear(i):
        if i == len(dims) - 2:
            layer = [torch.nn.Linear(dims[i], dims[i + 1])]
            if layernorm:
                layer.insert(0, torch.nn.LayerNorm(dims[i]))
            if dropout > 0:
                layer.insert(0, torch.nn.Dropout(dropout))
            return layer
        else:
            layer = [torch.nn.Linear(dims[i], dims[i + 1]), nonlinearity()]
            if layernorm:
                layer.insert(0, torch.nn.LayerNorm(dims[i]))
            if dropout > 0:
                layer.insert(0, torch.nn.Dropout(dropout))
            return layer

    modules = [linear(i) for i in range(len(dims) - 1)]
    if end_with is not None:
        modules.append([end_with()])
    modules = sum(modules, [])
    return torch.nn.Sequential(*modules)

def cast_dict_to_tensors(d, device="cpu"):
    if isinstance(d, dict):
        return {k: cast_dict_to_tensors(v, device) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float().to(device)
    elif isinstance(d, torch.Tensor):
        return d.float().to(device)
    else:
        return d

def cast_dict_to_numpy(d):
    if isinstance(d, dict):
        return {k: cast_dict_to_numpy(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.detach().cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return d
    

class RunningMaxMin():
    def __init__(self):
        super().__init__()
        self.max = None
        self.min = None

    def forward(self, x):
        x = rearrange(x, '... d -> b d')
        if self.max is None:
            self.max = torch.max(x, dim=0)
            self.min = torch.min(x, dim=0)
        else:
            curr_max = torch.max(x, dim=0)
            self.max = torch.maximum(self.max, curr_max)
            curr_min = torch.min(x, dim=0)
            self.min = torch.minimum(self.min, curr_min)

class BinaryMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, layers_n, **kwargs):
        super().__init__()
        next_pow_two = lambda x: 2**(x - 1).bit_length()
        mid_dims = np.linspace(in_dim, out_dim, layers_n, dtype=int).tolist()[1:-1]
        mid_dims = list(map(next_pow_two, mid_dims))
        self.layer0 = torch.nn.Sequential(torch.nn.Linear(in_dim, in_dim), torch.nn.Sigmoid())
        self.layers = sequential([in_dim] + mid_dims + [out_dim], **kwargs)

    def forward(self, x):
        x_soft = self.layer0(x)
        x_hard = (x_soft > 0.5).float()
        return self.layers(x_hard - x_soft.detach() + x_soft)

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, layers_n, **kwargs):
        super().__init__()
        next_pow_two = lambda x: 2**(x - 1).bit_length()
        mid_dims = np.linspace(in_dim, out_dim, layers_n, dtype=int).tolist()[1:-1]
        mid_dims = list(map(next_pow_two, mid_dims))
        self.layers = sequential([in_dim] + mid_dims + [out_dim], **kwargs)

    def forward(self, x):
        return self.layers(x)

def freeze(model) -> None:
    r"""
    Freeze all params for inference.
    """
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

def load_model(model_path: str, cls: str = None, **kwargs) -> LightningModule:
    """"
    load a pytorch lightning model from model_path
    if cls is not given, try to find the class in the config
    """
    if cls is None:
        d = torch.load(model_path)
        cls = d['hyper_parameters']['cfg']['model']['_target_']
    cls: LightningModule = get_class(cls)
    return cls.load_from_checkpoint(model_path, **kwargs)

def generate_id(resume_hash=None) -> str:
    # ~3t run ids (36**8)
    if resume_hash is None:
        run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
        return run_gen.random(8)
    else:
        return resume_hash

def goals2plan(goals: Tensor, device='cpu', duration: float = 1.0):
    waypoints = goals[:, :3]
    if goals.shape[-1] == 4:
        mdurations = goals[:, 3]
    else:
        mdurations = torch.ones_like(goals[:, 0]) * duration
    mframes = (mdurations * 30).int()
    goal_loc = torch.repeat_interleave(waypoints, mframes, dim=0)
    goal_eta = torch.cat([torch.flip(torch.arange(1, f+1), dims=[0]) for f in mframes], dim=0)
    # goal_eta = torch.clamp(goal_eta-60, min=10)  # thwma larwse
    return goal_loc[:, None].to(device), goal_eta[:, None, None].to(device)

def goal2plan(goals: Tensor, frames: int):
    """
    goals: (b, 3) Tensor with goal locations
    """
    goal_loc = repeat(goals, 'b ... -> f b ...', f=frames)
    goal_eta = torch.flip(torch.arange(1, frames+1), dims=[0])
    goal_eta = repeat(goal_eta, 'f -> f b 1', b=goals.shape[0])
    # subtract 30 frames and clamp to 30
    # goal_eta = torch.clamp(goal_eta - 90, min=5)
    return goal_loc, goal_eta