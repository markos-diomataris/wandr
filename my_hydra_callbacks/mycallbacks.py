import logging
import pickle
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback


class MyPickleJobInfoCallback(Callback):
    """Pickle the job config/return-value in ${output_dir}/{config,job_return}.pickle"""

    output_dir: Path

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Pickle the job's config in ${output_dir}/config.pickle."""
        self.output_dir = Path(config.hydra.runtime.output_dir) / Path(
            config.hydra.output_subdir
        )
        filename = "config.pickle"
        if Path.exists(self.output_dir / filename):
            self.log.info(f"NOT saving job configs, already found in {self.output_dir / filename}")
        else:
            self._save_pickle(obj=config, filename=filename, output_dir=self.output_dir)
            self.log.info(f"Saving job configs in {self.output_dir / filename}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Pickle the job's return value in ${output_dir}/job_return.pickle."""
        filename = "job_return.pickle"
        self._save_pickle(obj=job_return, filename=filename, output_dir=self.output_dir)
        self.log.info(f"Saving job_return in {self.output_dir / filename}")

    def _save_pickle(self, obj: Any, filename: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None
        with open(str(output_dir / filename), "wb") as file:
            pickle.dump(obj, file, protocol=4)