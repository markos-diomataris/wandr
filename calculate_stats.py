import hydra
import logging

from einops import rearrange
import torch
from omegaconf import DictConfig
import logging
import random
from os.path import exists, join
from pathlib import Path
from hydra.utils import instantiate

import joblib
import numpy as np
import torch
from utils.misc import cast_dict_to_tensors
from pdb import set_trace
from tqdm import tqdm

from dataset.amass_dataset import AmassDataset


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="calculate_stats.yaml")
def main(cfg : DictConfig) -> None:
    data_dict = {}
    for path in cfg.load_files: 
        log.info(f"Loading {path}")
        data_dict = cast_dict_to_tensors(joblib.load(path)) | data_dict
    if len(iter(data_dict.values()).__next__()['rots'].shape) == 3:
        for k, v in data_dict.items():
            data_dict[k]['rots'] = rearrange(v['rots'], 'f j d -> f (j d)')
    dataset = instantiate(cfg.dataset, data=[v for k, v in data_dict.items()],
                          do_augmentations=False)
    # dataset = AmassDataset([v for k, v in data_dict.items()], cfg,
    #                        do_augmentations=False)
    # if not exists(stat_path) or cfg.dl.force_recalculate_stats:
    if not exists(cfg.statistics_file):
        log.info(f"No dataset stats found. Calculating and saving to {cfg.statistics_file}")
    else:
        log.info(f"Dataset stats will be re-calculated and saved to {cfg.statistics_file}")
    feature_names = dataset._feat_get_methods.keys()
    feature_dict = {name: [] for name in feature_names}
    for i in tqdm(range(len(dataset))):
        x = dataset.get_all_features(i)
        for name in feature_names:
            feature_dict[name].append(x[name])
    feature_dict = {name: torch.cat(feature_dict[name], dim=0) for name in feature_names}
    stats = {name: {'max': x.max(0)[0].numpy(),
                    'min': x.min(0)[0].numpy(),
                    'mean': x.mean(0).numpy(),
                    'std': x.std(0).numpy()}
                for name, x in feature_dict.items()}
    log.info("Calculated statistics for the following features:")
    log.info(feature_names)
    log.info(f"saving to {cfg.statistics_file}")
    np.save(cfg.statistics_file, stats)
    log.info("Done.")

if __name__=='__main__':
    main()
