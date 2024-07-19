import os
from omegaconf import DictConfig, OmegaConf
import trimesh
from tqdm import tqdm
import pandas as pd

from utils.misc import cast_dict_to_tensors, cast_dict_to_numpy, generate_id
import hydra
import numpy as np
import torch
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
import wandb
import logging
import joblib
from dataset.goal_dataset import CylinderDataModule

from models.human import Human
from rendering.render_utils import render_motion
from utils.misc import generate_id, load_model, cast_dict_to_numpy, goal2plan

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("generate_id", generate_id, use_cache=True)


@hydra.main(version_base=None, config_path="conf", config_name="evaluate.yaml")
def evaluate(cfg: DictConfig) -> None:
    # log job hash
    log.info(f"Run hash: {cfg.run_hash}")

    # init wandb if it is not None
    if cfg.logger.mode is not None:
        logger = wandb.init(**cfg.logger,
                            config=OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = None

    if cfg.load_model_from_artifact:
        # use wandb api to download the model artifact
        api = wandb.Api()
        artifact = api.artifact(cfg.model_artifact, type='model')
        artifact_dir = artifact.download('./')
        model_path = os.path.join(artifact_dir, "model.ckpt")
    else:
        model_path = cfg.model_path
    # load the model
    model: Human = Human.load_from_checkpoint(model_path)
    model.freeze()
    model.cuda()

    # delete the model file if we loaded it from an artifact
    if cfg.load_model_from_artifact:
        os.remove('./model.ckpt')
 
    # create output directory
    os.makedirs('./output_data', exist_ok=True)
    os.makedirs('./output_data/meshes', exist_ok=True)

    ###############
    ### RENDERER ##
    ###############
    if cfg.rendering.render:
        from aitviewer.configuration import CONFIG as C
        os.system("Xvfb :11 -screen 0 640x480x24 &")
        os.environ['DISPLAY'] = ":11"
        C.update_conf({"playback_fps": 30,
                       "auto_set_floor": False,
                       "z_up": True})
        renderer = HeadlessRenderer()
    else:
        renderer = None

    # setup evaluator
    evaluator = instantiate(cfg.evaluator)

    #######################
    ### DATALOADER ########
    #######################
    data_module = instantiate(cfg.dataloader,
                              shuffle_datasets=[False, False, False],
                              _recursive_=False)
    # grab an initial state from the test set
    data_module.setup('Beethoven sonata F minor')
    dataset = data_module.dataset['test']
    if cfg.pick_random_idx:
        idx = np.random.randint(0, len(dataset))
    else:
        idx = cfg.data_idx
    print(idx)
    batch = dataset[idx]
    goal_datamodule = instantiate(cfg.goal_dataloader,
                                  dataset={"init_orient": batch["body_orient"][0],
                                           "init_pelvis": batch["body_joints"][0, :3]})
    if logger is not None:
        # log batch filename with wandb
        logger.log({"filename": batch['filename']})
        logger.log({"data_idx": idx})

    frames = int(cfg.motion_duration * 30)
    with torch.no_grad():
        i=-1
        for goal_position, radius, theta, height in tqdm(goal_datamodule.test_dataloader()):
            i += 1
            goal = goal2plan(goal_position, frames=frames)
            goal = goal[0].cuda(), goal[1].cuda()
            motion_batch = data_module.collate_fn([batch] * goal[0].shape[1])
            motion_batch = cast_dict_to_tensors(motion_batch, model.device)
            motion, out, joints, vertices = model.rollout(motion_batch, n_steps=frames-1,
                                                sigma_mult=1.0,
                                                latents=None,
                                                goal=goal,
                                                fwd_smpl=True,
                                                fast=False,
                                                return_vertices=True)

            # also add the goal, joints and vertices to the motion
            motion['goal'] = goal[0]
            motion['joints'] = joints
            motion['vertices'] = vertices
            motion_numpy = cast_dict_to_numpy(motion)
            evaluator.evaluate_motion_batch(motion_numpy,
                                            meta_data={'radius': radius,
                                                       'theta': theta,
                                                       'height': height})
            # if cfg.save_meshes:
            #     log.info("Saving meshes to ./output_data/meshes")
            #     for f in range(vertices.shape[0]):
            #         mesh = trimesh.Trimesh(vertices=vertices[f, 0], faces=model.body_model.faces)
            #         mesh.export(f"./output_data/meshes/{f}.obj")

            # # save the motion
            # joblib.dump(motion_numpy, './output_data/motion{i}.joblib')

            # if cfg.rendering.render:
            #     video_name = f"./output_data/motion{i}.mp4"
            #     render_motion(renderer, cast_dict_to_numpy(motion),
            #                 filename=video_name,
            #                 pose_repr="6d",
            #                 camera_lock_offset=(-2, -2, 2))
            #     if logger is not None:
            #         wandb.log({"renders/motion": wandb.Video(video_name)})
            # using wandb log the output dir as artifact
            # if logger is not None:
            #     wandb.save('./output_data/*')
        results = evaluator.get_metrics()
        results_dict = results['metrics_avg'] | results['meta_data'] | results['metrics']
        # turn results_dict into panda dataframe
        results_df = pd.DataFrame.from_dict(results_dict)
        # save as csv
        results_df.to_csv('./output_data/results.csv')
        # if there is a wandb logger, then log it as a table
        if logger is not None:
            wandb.log({"results": wandb.Table(dataframe=results_df)})
            wandb.log({'metrics': results['metrics_avg']})
        # print only the average results  as pandas dataframe
        print(results['metrics_avg'])
        

    
 
if __name__ == "__main__":
    evaluate()
