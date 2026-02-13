"""
Script for running the experiments.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import random
from time import sleep
import subprocess

import hydra

os.environ["WANDB_MODE"] = "disabled"
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import utils.hydra  # DONT REMOVE THIS!!
from omegaconf import DictConfig, OmegaConf

# from utils.wandb_utils import merge_wandb_cfg

# from neuralop.training import Trainer, setup
from src.wos_trainer import WoSTrainer
from src.losses import LinearPoissonEqnLoss, DeepRitzLoss
from neuralop.losses import LpLoss
from neuralop.utils import count_model_params

from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    sleep(random.randint(1, 10))
    print(f"cfg: {cfg}")
    logging.info("---------------------------------------------------------------")
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    # Log overrides
    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    out_dir = Path(hydra_config.runtime.output_dir).absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("out_dir"):
        cfg.out_dir = str(out_dir)

    # Setup wandb
    tags = [t for t in hydra_config.overrides.task if len(t) < 32]
    if "wandb" not in cfg:
        cfg.wandb = OmegaConf.create()
    if not cfg.wandb.get("tags"):
        cfg.wandb.tags = tags

    # We will group experiments by their core components, like model and loss function.
    if not cfg.wandb.get("group"):
        # This assumes your config structure is like `cfg.model.name` and `cfg.loss.name`
        # Adjust the path if your config is structured differently.
        try:
            cfg.wandb.group = f"{cfg.model.name}-{cfg.loss.name}"
            logging.info(f"Automatically setting wandb group to: {cfg.wandb.group}")
        except Exception as e:
            logging.warning(f"Could not auto-generate wandb group name: {e}")
            cfg.wandb.group = "default_group"

    if not cfg.wandb.get("id"):
        # create id based on log directory for automatic (slurm) resuming
        sha = hashlib.sha256()
        sha.update(str(out_dir).encode())
        cfg.wandb.id = sha.hexdigest()

    if not cfg.wandb.get("name"):
        if hydra_config.mode is hydra.types.RunMode.RUN:
            name = str(out_dir.relative_to(out_dir.parents[1]))
        else:
            name = str(out_dir.parent.relative_to(out_dir.parents[2]))
        cfg.wandb.name = name + "," + ",".join([t.split("=")[-1] for t in tags])

    OmegaConf.set_struct(cfg, True)
    if cfg.logger.get("log", True):
        wandb.init(
            dir=out_dir,
            settings=wandb.Settings(console="off"),
            **cfg.wandb,
        )
        # Define a custom x-axis for cleaner epoch-level plots, as recommended before.
        logging.info("Defining custom wandb metrics for epoch-based plotting.")
        wandb.define_metric("epoch")
        wandb.define_metric("epoch_step")
        # wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch_step")
        # You can also define one for batch-level metrics if you use them
        wandb.define_metric("global_step")
        wandb.define_metric("train/batch_*", step_metric="global_step")
        # os.environ["WANDB_MODE"] = "disabled"

        # Resume old wandb run
        # if wandb.run is not None and wandb.run.resumed:
        #     logging.info("Resume wandb run %s", wandb.run.path)
        #     if cfg.get("merge_wandb_resume_cfg"):
        #         cfg = merge_wandb_cfg(cfg)

        # Log config and overrides
        logging.info("---------------------------------------------------------------")
        logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
        logging.info("---------------------------------------------------------------")

        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_config["hydra"] = OmegaConf.to_container(hydra_config, resolve=True)

        for k in [
            "help",
            "hydra_help",
            "hydra_logging",
            "job_logging",
            "searchpath",
            "callbacks",
            "sweeper",
        ]:
            wandb_config["hydra"].pop(k, None)
        wandb.config.update(wandb_config, allow_val_change=True)

        # run solver
        logging.info("---------------------------------------------------------------")
    else:
        cfg.wandb.mode = "disabled"

    try:
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model)
        train_loss = instantiate(cfg.loss)

        n_params = count_model_params(model)

        if cfg.wandb.mode != "disabled":
            wandb.log({"n_params": n_params})
            wandb.watch(model)
        # dataset
        train_dataset = instantiate(cfg.dataset, isTrain=True)
        test_dataset = instantiate(cfg.dataset, isTrain=False)

        logging.info("Train dataset: %s", train_dataset)
        logging.info("Test dataset: %s", test_dataset)
        train_loader = DataLoader(train_dataset)
        test_loaders = {
            "test": DataLoader(test_dataset),
        }
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

        trainer = WoSTrainer(
            model=model,
            n_epochs=cfg.train.n_epochs,
            data_processor=None,
            device=cfg.train.device,
            eval_interval=cfg.logger.eval_interval,
            log_output=cfg.logger.log_output,
            use_distributed=cfg.get("use_distributed", False),
            verbose=cfg.logger.verbose,
            wandb_log=cfg.wandb.get("mode", "disabled") != "disabled",
        )

        l2loss = LpLoss(d=2, p=2)

        def mse_loss(y_pred, y, **kwargs):
            assert y_pred.shape == y.shape, (y.shape, y_pred.shape)
            return ((y_pred - y) ** 2).mean()

        eval_losses = {
            "mse": mse_loss,
            "l2": l2loss,
        }
        print(f"Number of model parameters: {n_params}")
        trainer.train(
            train_loader,
            test_loaders,
            optimizer,
            scheduler,
            regularizer=False,
            training_loss=train_loss,
            eval_losses=eval_losses,
        )

        wandb.run.summary["error"] = None
        logging.info("Completed ✅")
        wandb.finish()

    except Exception as e:
        logging.critical(e, exc_info=True)
        wandb.run.summary["error"] = str(e)
        wandb.finish(exit_code=1)


def sync_wandb(wandb_dir: Path | str):
    run_dirs = [f for f in Path(wandb_dir).iterdir() if "run-" in f.name]
    for run_dir in sorted(run_dirs, key=os.path.getmtime):
        logging.info("Syncing %s.", run_dir)
        subprocess.run(
            ["wandb", "sync", "--no-include-synced", "--mark-synced", str(run_dir)]
        )


if __name__ == "__main__":
    main()
