import json
from smplx.joint_names import JOINT_NAMES
from typing import List, Optional

class AmassFilter:
    def __init__(self, file_paths: Optional[List[str]]=None,
                 feet_height_threshold: Optional[float]=None,
                 exclude_grab: Optional[bool]=False):
        self.blacklist = []
        self.feet_height_threshold  = feet_height_threshold
        self.exclude_grab = exclude_grab
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                files = json.load(f)
                files = [f.replace("_stageii", "").replace("_poses", "")
                            for f in files]
                # split by / and get last two elements
                files = ["/".join(f.split("/")[-2:]) for f in files]
                self.blacklist.extend(files)

    def is_good_quality(self, data):
        if len(self.blacklist) > 0:
            if self.fname_exist_in_blacklist(data['fname']):
                # print(f'fname {data["fname"]} will be skipped: blacklisted')
                return False
        if self.feet_height_threshold is not None:
            if self.feet_are_lifted_high(data['joint_positions']):
                # print(f'fname {data["fname"]} will be skipped: feet dz is > {self.feet_height_threshold}')
                return False
        if self.exclude_grab:
            if 'GRAB' in data['fname']:
                # print(f'fname {data["fname"]} will be skipped: exclude GRAB')
                return False
        
        return True

    def fname_exist_in_blacklist(self, fname):
        # remove "_stageii" or "_poses" from the fname
        fname = fname.replace("_stageii", "").replace("_poses", "")
        # split by / and get last two elements
        fname = "/".join(fname.split("/")[-2:])
        return fname in self.blacklist

    def feet_are_lifted_high(self, joints):
        # the feet never lie: https://www.youtube.com/watch?v=DUT5rEU6pqM&ab_channel=shakiraVEVO
        # get the foot joints indices named left/right_heel and left/right_foot
        feet_joint_names = ['left_foot', 'right_foot']
        feet_idx = tuple(JOINT_NAMES.index(n) for n in feet_joint_names)
        feet_joints = joints[:, feet_idx, :]
        # frames x 2 x 3
        min_z = feet_joints[..., 2].min()
        max_z = feet_joints[..., 2].max()
        max_diff = max_z - min_z
        return max_diff > self.feet_height_threshold
