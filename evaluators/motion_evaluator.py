from typing import List
from smplx.joint_names import JOINT_NAMES
from utils.misc import cast_dict_to_numpy
import numpy as np
from einops import reduce

class MotionEvaluator:
    def __init__(self, metrics_to_eval: List[str],
                 skating_threshold_height: float=0.05,
                 reach_threshold: float=0.10,
                 reached_policy: str='any'):
        self.metrics_to_eval = metrics_to_eval
        self.skating_threshold_height = skating_threshold_height
        self.reach_threshold = reach_threshold
        assert reached_policy in ['last', 'any']
        self.reached_policy = reached_policy
        self.eval_functions = {
            # 'foot_skating': self.calculate_foot_skating,
            'foot_skating_percentage': self.calculate_foot_skating_percentage,
            'goal_reached': self.calculate_if_goal_reached,
            'goal_distance': self.calculate_distance_to_goal,
            'wrist_velocity': self.calculate_wrist_velocity
        }
        self.metrics_batch = []
        self.meta_data = []

    def reset(self):
        self.metrics_batch = []
        self.meta_data = []
 
    def evaluate_motion_batch(self, motion: dict,
                              meta_data: dict=None):
        motion = cast_dict_to_numpy(motion)
        metrics = {metric: self.eval_functions[metric](motion) 
                   for metric in self.metrics_to_eval}
        self.metrics_batch.append(metrics)
        self.meta_data.append(meta_data)
        return metrics

    def get_raw_avg_metrics(self):
        metrics = {metric: np.concatenate([m[metric] for m in self.metrics_batch],
                                          axis=0)
                   for metric in self.metrics_to_eval}
        metrics_avg = {metric+'_avg': np.nanmean(np.concatenate([m[metric] for m in self.metrics_batch],
                                          axis=0))
                       for metric in self.metrics_to_eval}
        return {'metrics': metrics,
                'metrics_avg': metrics_avg}

    def get_metrics(self):
        metrics = {metric: np.concatenate([m[metric] for m in self.metrics_batch],
                                          axis=0)
                   for metric in self.metrics_to_eval}
        metrics_avg = {metric+'_avg': np.nanmean(np.concatenate([m[metric] for m in self.metrics_batch],
                                          axis=0))
                       for metric in self.metrics_to_eval}
        meta_data = {k: np.concatenate([m[k] for m in self.meta_data],
                                       axis=0)
                     for k in self.meta_data[0].keys()}
        return {'metrics': metrics,
                'metrics_avg': metrics_avg,
                'meta_data': meta_data}

    def calculate_distance_to_goal(self, motion):
        """
        return the distance to the goal for the last frame of the motion
        """
        # get right wrist index from smplx joints
        rwrist_idx = JOINT_NAMES.index('right_wrist') 
        right_wrist = motion['joints'][..., rwrist_idx, :]
        # check the distance between the right wrist and the goal
        if self.reached_policy == 'last':
            # for the last frame of the motion
            distance = np.linalg.norm(right_wrist[-1] - motion['goal'][-1], axis=-1)
        elif self.reached_policy == 'any':
            # for any frame of the motion
            distance = np.linalg.norm(right_wrist - motion['goal'][-1:], axis=-1)
            distance = reduce(distance, 's b ... -> b ...', 'min')
        return distance

    def calculate_if_goal_reached(self, motion):
        """
        return 1 if the goal is reached, 0 otherwise
        """
        # get right wrist index from smplx joints
        rwrist_idx = JOINT_NAMES.index('right_wrist') 
        right_wrist = motion['joints'][..., rwrist_idx, :]
        # check if the right wrist is within a threshold distance from the goal
        if self.reached_policy == 'last':
            # for the last frame of the motion
            reached = np.linalg.norm(right_wrist[-1] - motion['goal'][-1], axis=-1) <= self.reach_threshold
        elif self.reached_policy == 'any':
            # for any frame of the motion
            reached = np.linalg.norm(right_wrist - motion['goal'][-1:], axis=-1) <= self.reach_threshold
            reached = reached.any(axis=0)
        return reached * 1

    def calculate_wrist_velocity(self, motion):
        """
        calculate the velocity of the right wrist when the goal is reached
        """
        # get right wrist index from smplx joints
        rwrist_idx = JOINT_NAMES.index('right_wrist') 
        right_wrist = motion['joints'][..., rwrist_idx, :]
        # check the distance between the right wrist and the goal
        # claculate the frame where the goal is reached
       # Calculate the distance between the right wrist and the goal
        distance_to_goal = np.linalg.norm(right_wrist - motion['goal'][-1:], axis=-1)
        reached = distance_to_goal <= self.reach_threshold
        when_reached = reached.argmax(axis=0)

        # Find the index of the frame closest to the goal
        # closest_frame_index = np.argmin(distance_to_goal, axis=0)
        # Calculate the difference in right wrist positions
        wrist_velocities = np.linalg.norm(np.diff(right_wrist, axis=0) * 30, axis=-1)
        # cilip so that idx is not out of bounds
        # closest_frame_index = np.clip(closest_frame_index, 0, wrist_velocities.shape[0]-1)
        when_reached = np.clip(when_reached, 0, wrist_velocities.shape[0]-1)
        reach_velocities = wrist_velocities[when_reached, range(0,when_reached.shape[0])]
        have_reached = reached[when_reached, range(0,when_reached.shape[0])]
        reach_velocities[~have_reached] = np.nan
        return reach_velocities

    def calculate_foot_skating_percentage(self, motion):
        """
        Calculate the percentage of frames where the lowest vertex of the mesh
        has a velocity above 1cm per frame

                            !! in Circle they use 20fps !!
        """
        vertices = motion['vertices']
        # get lowest vertex
        min_vertex_idx = vertices[:-1, ..., 2].argmin(axis=-1)
        # calculate min_vertex velocity
        vertex_diff = np.diff(vertices[..., :2], axis=0)
        # get only velocities of the minimum vertex that are above a threshold
        i, j = np.indices(min_vertex_idx.shape)
        min_vertex_diff = vertex_diff[i, j, min_vertex_idx]
        # get the norm of diffs
        min_vertex_diff_norm = np.linalg.norm(min_vertex_diff, axis=-1)
        return (min_vertex_diff_norm > ((2/3)*0.01)).sum(0) / min_vertex_diff_norm.shape[0]
    
    def calculate_foot_skating(self, motion):
        """
        Calculate foot skating based on feet joints
        its the same as calculate_foot_skating_percentage
        """
        vertices = motion['joints']
        # get lowest vertex
        min_vertex_idx = vertices[:-1, ..., 2].argmin(axis=-1)
        # calculate min_vertex velocity
        vertex_diff = np.diff(vertices[..., :2], axis=0)
        # get only velocities of the minimum vertex that are above a threshold
        i, j = np.indices(min_vertex_idx.shape)
        min_vertex_diff = vertex_diff[i, j, min_vertex_idx]
        # get the norm of diffs
        min_vertex_diff_norm = np.linalg.norm(min_vertex_diff, axis=-1)
        return (min_vertex_diff_norm > ((2/3)*0.01)).sum(0) / min_vertex_diff_norm.shape[0]