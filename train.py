import hydra
import logging
import os

import torch
import pickle
from omegaconf import DictConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.misc import generate_id

torch.multiprocessing.set_sharing_strategy('file_system')

log = logging.getLogger(__name__)

# Register custom resolvers
OmegaConf.register_new_resolver("generate_id", generate_id, use_cache=True)
# OmegaConf.register_new_resolver("merge", lambda x, y : x + y)

@hydra.main(version_base=None, config_path="conf", config_name="train.yaml")
def main(cfg : DictConfig) -> None:

    ####################
    ### SETUP LOGGING ##
    ####################
    if cfg.resume_hash is not None:
        resume_hash_cache = cfg.resume_hash
        cfg.run_hash = cfg.resume_hash
        with open('.hydra/config.pickle', 'rb') as fd:
            old_cfg = pickle.load(fd)
        cfg = old_cfg  # merge
        cfg.resume_hash = resume_hash_cache
        log.info('Using saved config.')
    log.info(f"Run hash: {cfg.run_hash}")
    if cfg.wandb_logger.mode  == 'offline':
        logger = None
        log.info(f"Will not be logging with wandb.")
    else:
        model_name = cfg.model._target_.split('.')[-1]
        logger = instantiate(cfg.wandb_logger, name=f"{model_name}-{cfg.run_hash}")
                            #  settings=wandb.Settings(start_method="fork"))
        if cfg.resume_hash is None:
            # TODO: this should not need an 'if' statement
            logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = []
    ################################
    ### SETUP MODEL CHECKPOINTING ##
    ################################
    callbacks.append(instantiate(cfg.model_checkpoint))
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    ######################
    ### OTHER CALLBACKS ## 
    ######################

    #######################
    ### DATALOADER ########
    #######################
    data_module = instantiate(cfg.dataloader, _recursive_=False)

    ##############
    ### TRAINER ##
    ##############
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    ###############
    ### RENDERER ##
    ###############
    if cfg.rendering.in_training or cfg.rendering.in_testing:
        from aitviewer.headless import HeadlessRenderer
        from aitviewer.configuration import CONFIG as C
        os.system("Xvfb :11 -screen 0 640x480x24 &")
        os.environ['DISPLAY'] = ":11"
        C.update_conf({"playback_fps": 30,
                       "auto_set_floor": False,
                       "z_up": True,
                       "smplx_models": cfg.paths.smplx_models_path})
        renderer = HeadlessRenderer()
    else:
        renderer = None

    ############
    ### MODEL ##
    ############
    model = instantiate(cfg.model, renderer=renderer, _recursive_=False)

    #################
    ### TRAIN/TEST ##
    #################
    resume_path = None
    if cfg.resume_hash is not None:
        resume_path = os.path.join(cfg.paths.chkpnt_dir, 'last.ckpt')
        resume_path = resume_path if os.path.exists(resume_path) else None
    if cfg.watch_model:
        logger.watch(model, log=cfg.log, log_freq=cfg.log_freq)
    trainer.fit(model, data_module, ckpt_path=resume_path)
    # trainer.test(model, data_module)

if __name__ == "__main__":
    main()