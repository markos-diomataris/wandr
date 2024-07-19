import logging
import random
from utils.transformations import flip_motion

from utils.misc import DotDict, subsample, cut_chunk


# A logger for this file
log = logging.getLogger(__name__)

class SequenceParserAmass:

    def __init__(self, framerate_ratio: int=1, chunk_duration=None,
                 random_chunk=True, random_flip=False):
        self.framerate_ratio = framerate_ratio
        self.chunk_duration = chunk_duration
        self.random_chunk = random_chunk
        self.random_flip = random_flip
        self._filter_methods = [
            # self._framerate_subsample,
        ]
        self._augment_methods = [
            self._cut_chunk,
            self._flip_motion,
            # self._random_rotate,
            # self._trim_nointeractions,
        ]


    def parse_datum(self, datum: dict):
        datum['chunk_start'] = 0
        datum['n_frames'] = datum['rots'].shape[0]

        # filter sequence with various methods
        # !! Be careful of the order in config, it can make a difference !!
        for filter_fn in self._filter_methods:
            datum = filter_fn(datum)
        
        return DotDict(datum)

    def augment_npz(self, npz):
        for augment_fn in self._augment_methods:
            npz = augment_fn(npz)
        return DotDict(npz)

     ### PREPROCESSING METHODS ###  

    def _framerate_subsample(self, npz):
        if self.framerate_ratio is not None:
            assert isinstance(self.framerate_ratio, int)
            # reduce framerate
            framerate_ratio = self.framerate_ratio
            old_framerate = npz['framerate']
            new_framerate = old_framerate / framerate_ratio
            npz = subsample(npz, framerate_ratio)
            npz['framerate'] = new_framerate
            npz['n_frames'] = npz['body']['params']['transl'].shape[0]
        return DotDict(npz)

    ### AUGMENTATION METHODS | only done on train/val-set ###  

    def _flip_motion(self, data):
        # flip motion 50% of the time
        if self.random_flip and random.random() < 0.5:
            return DotDict(flip_motion(data))
        else:
            return DotDict(data)

    def _cut_chunk(self, data):
        # cut to smaller continuous chunks
        if self.chunk_duration is not None:
            chunk_length = min(int(self.chunk_duration * data['fps']),
                               data['n_frames'])
            if self.random_chunk:
                chunk_start = random.randint(0, data['n_frames'] - chunk_length)
            else:
                chunk_start = 0
            data = cut_chunk(data, chunk_start, chunk_length)
        return DotDict(data)
    

    # def _random_rotate(self, npz):
    #     if self.cfg.aug.random_rotate:
    #         # print("randomly rotated")
    #         if self.cfg.aug.random_rot_type == "3d":
    #             rot_mat = roma.random_rotmat()
    #         elif self.cfg.aug.random_rot_type == "z_axis":
    #             theta = torch.rand(1) * 2 * np.pi
    #             rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
    #                                    [torch.sin(theta), torch.cos(theta), 0],
    #                                    [0, 0, 1]])
    #         npz['joints'] = torch.einsum('fjk,kd->fjd', torch.from_numpy(npz['joints']), rot_mat).numpy()
    #         pelvis_orient = torch.from_numpy(npz['body']['params']['global_orient'])
    #         pelvis_orient = axis_angle_to_matrix(pelvis_orient)
    #         npz['body']['params']['global_orient'] = matrix_to_axis_angle(
    #             torch.einsum('ij,fik->fjk', rot_mat, pelvis_orient)).numpy()
    #         npz['object']['params']['transl'] = torch.einsum('fk,kd->fd', torch.from_numpy(npz['object']['params']['transl']).float(), rot_mat).numpy()
    #     return DotDict(npz)

