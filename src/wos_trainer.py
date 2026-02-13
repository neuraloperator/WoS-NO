from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings
import time
import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from collections import defaultdict

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from neuralop.training.training_state import load_training_state, save_training_state


class WoSTrainer:
    """
    A general Trainer class to train neural-operators on given datasets.

    .. note ::
        Our Trainer expects datasets to provide batches as key-value dictionaries, ex.:
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses.
        For specifics and an example, check ``neuralop.data.datasets.DarcyDataset``.

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str 'cpu' or 'cuda'
    mixed_precision : bool, default is False
        whether to use torch.autocast to compute mixed precision
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use DDP
    verbose : bool, default is False
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: nn.Module = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
    ):
        """ """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = wandb_log and wandb.run is not None
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        # handle autocast device
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor

        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def _init_cache(self, train_loader):
        self.cache = [0 for _ in range(len(train_loader))]  # cache for wos estimates
        self.highest_loss = 1e10
        self.error_wrt_gt = []
        self.error_wrt_fenics = []

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        eval_modes=None,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        max_autoregressive_steps: int = None,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here.

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        eval_modes: dict[str], optional
            optional mapping from the name of each loader to its evaluation mode.

            * if 'single_step', predicts one input-output pair and evaluates loss.

            * if 'autoregressive', autoregressively predicts output using last step's
            output as input for a number of steps defined by the temporal dimension of the batch.
            This requires specially batched data with a data processor whose ``.preprocess`` and
            ``.postprocess`` both take ``idx`` as an argument.
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model,
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        max_autoregressive_steps : int, default None
            if provided, and a dataloader is to be evaluated in autoregressive mode,
            limits the number of autoregressive in each rollout to be performed.

        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders

        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. This means that the loss is "
                    "initialized to average across the batch dim. The Trainer "
                    "expects losses to sum across the batch dim."
                )

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # create default eval modes
        if eval_modes is None:
            eval_modes = {}

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            self.model = DDP(
                self.model, device_ids=[device_id], output_device=device_id
            )

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert (
                self.save_best in metrics
            ), f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float("inf")
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f"Training on {len(train_loader.dataset)} samples")
            print(
                f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}."
            )
            sys.stdout.flush()

        self._init_cache(train_loader)

        for epoch in range(self.start_epoch, self.n_epochs):
            (
                train_err,
                avg_loss,
                avg_lasso_loss,
                avg_loss_wrt_gt,
                avg_wos_loss_wrt_gt,
                epoch_train_time,
            ) = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
                avg_loss_wrt_gt=avg_loss_wrt_gt,
                avg_wos_loss_wrt_gt=avg_wos_loss_wrt_gt,
            )

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    n_batch=len(train_loader),
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        training_loss : training.losses function
            cost function to minimize

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_loss_wrt_gt = 0
        avg_lasso_loss = 0
        avg_wos_loss_wrt_gt = 0

        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0

        # track number of training examples in batch
        self.n_samples = 0
        prev_idx = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", file=sys.stdout)
        n_batches = len(train_loader)
        mse = torch.nn.MSELoss()
        epoch_metrics = defaultdict(float)
        wos_loss = 0.0
        for idx, sample in enumerate(progress_bar):

            loss, gt, out, sample = self.train_one_batch(
                idx, sample, training_loss, epoch
            )
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            progress_bar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Train Err: ": mse(out, sample["y"]).item(),
                    "avg_err": avg_loss / (idx + 0.00003),
                }
            )

            global_step = epoch * n_batches + idx
            if self.wandb_log:
                batch_metrics = {
                    "train/batch_loss": loss.item(),
                    "train/batch_mse_wos": mse(out, sample["y"]).item(),
                    "train/batch_avg_loss": avg_loss / (idx + 0.00003),
                }
                batch_metrics["global_step"] = global_step
                wandb.log(batch_metrics)  # Log against a global step counter
                # --- 2. Aggregate for Epoch-Level Logging ---
                epoch_metrics["epoch_loss"] += loss.item()
                epoch_metrics["epoch_mse_wos"] += mse(out, sample["y"]).item()

            if idx % 1 == 0 and idx > 0:
                self.error_wrt_gt.append(mse(out, sample["y"]).item())
                self.error_wrt_fenics.append(mse(out, gt.to(self.device)).item())
            if idx > 0:
                if (
                    avg_loss / idx < self.highest_loss
                    and avg_loss / idx > 0
                    and idx - prev_idx > 10
                ):
                    prev_idx = idx
                    self.highest_loss = avg_loss / idx
                    print(f"AVG LOSS IMPROVED {avg_loss/idx}: SAVING MODEL WEIGHTS")
            # if(idx %500== 0):
            #     mse = torch.nn.MSELoss()
            #     print("Epoch: ", epoch, "Step: ", idx, "Trn Err: ", loss.item(), "Train Err WOS: ", mse(out, sample['y']).item(),
            #             " Trn Err GT", avg_loss_wrt_gt/(idx+0.000004), " Avg Ls WOS: ", avg_loss/(idx+0.00003),  " Avg Ls GT: ", avg_loss_wrt_gt/(idx+0.00003), " WOS vs. FEN: ", mse(sample['y'], gt).item())
            with torch.no_grad():
                avg_loss += mse(out, sample["y"]).item()
                avg_loss_wrt_gt += mse(out, gt).item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        if self.wandb_log:
            # Average the metrics over all batches in the epoch
            final_epoch_metrics = {
                key: value / n_batches for key, value in epoch_metrics.items()
            }
            # wandb.log(final_epoch_metrics, step=global_step)
            # Add the epoch number itself for plotting
            final_epoch_metrics["epoch"] = epoch
            # Log the dictionary to wandb
            wandb.log(final_epoch_metrics)

        return (
            train_err,
            avg_loss,
            avg_lasso_loss,
            avg_loss_wrt_gt,
            avg_wos_loss_wrt_gt,
            epoch_train_time,
        )

    def evaluate_all(
        self,
        epoch,
        eval_losses,
        test_loaders,
        eval_modes,
        max_autoregressive_steps=None,
        n_batch=None,
    ):
        """evaluate_all iterates through the entire dict of test_loaders
        to perform evaluation on the whole dataset stored in each one.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_losses : dict[Loss]
            keyed ``loss_name: loss_obj`` for each pair. Full set of
            losses to use in evaluation for each test loader.
        test_loaders : dict[DataLoader]
            keyed ``loader_name: loader`` for each test loader.
        eval_modes : dict[str], optional
            keyed ``loader_name: eval_mode`` for each test loader.
            * If ``eval_modes.get(loader_name)`` does not return a value,
            the evaluation is automatically performed in ``single_step`` mode.
        max_autoregressive_steps : ``int``, optional
            if provided, and one of the test loaders has ``eval_mode == "autoregressive"``,
            limits the number of autoregressive steps performed per rollout.

        Returns
        -------
        all_metrics: dict
            collected eval metrics for each loader.
        """
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                epoch=epoch,
                n_batch=n_batch,
                log_prefix=loader_name,
                mode=loader_eval_mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def evaluate(
        self,
        loss_dict,
        data_loader,
        log_prefix="",
        epoch=None,
        n_batch=None,
        mode="single_step",
        max_steps=None,
    ):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        mode : Literal {'single_step', 'autoregression'}
            if 'single_step', performs standard evaluation
            if 'autoregression' loops through `max_steps` steps
        max_steps : int, optional
            max number of steps for autoregressive rollout.
            If None, runs the full rollout.
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # Ensure model and data processor are loaded to the proper device

        self.model = self.model.to(self.device)
        if (
            self.data_processor is not None
            and self.data_processor.device != self.device
        ):
            self.data_processor = self.data_processor.to(self.device)

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, "reduction"):
                if eval_loss.reduction == "mean":
                    warnings.warn(
                        f"{eval_loss.reduction=}. This means that the loss is "
                        "initialized to average across the batch dim. The Trainer "
                        "expects losses to sum across the batch dim."
                    )

        self.n_samples = 0
        with torch.no_grad():
            progress_bar = tqdm(
                data_loader, desc=f"Evaluation {epoch}", file=sys.stdout
            )
            avg_eval_loss = 0.0
            for idx, sample in enumerate(progress_bar):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                if mode == "single_step":
                    eval_step_losses, outs = self.eval_one_batch(
                        sample, loss_dict, return_output=return_output
                    )
                elif mode == "autoregression":
                    eval_step_losses, outs = self.eval_one_batch_autoreg(
                        sample,
                        loss_dict,
                        return_output=return_output,
                        max_steps=max_steps,
                    )
                avg_eval_loss += eval_step_losses["avg_loss_wrt_gt"]
                loss = eval_step_losses["avg_loss_wrt_gt"]
                progress_bar.set_postfix(
                    {"Loss": loss, "avg_loss": avg_eval_loss / (idx + 0.00003)}
                )
                global_step = idx
                if self.wandb_log:
                    batch_metrics = {
                        "val/batch_loss": loss,
                        "val/batch_avg_loss": avg_eval_loss / (idx + 0.00003),
                    }
                    batch_metrics["epoch_step"] = (epoch + 1) * n_batch
                    wandb.log(batch_metrics)  # Log against a global step counter
                for loss_name, val_loss in eval_step_losses.items():
                    if loss_name not in errors:
                        errors[f"{log_prefix}_{loss_name}"] = val_loss
                    else:
                        errors[f"{log_prefix}_{loss_name}"] += val_loss

        print(f"Eval: {log_prefix} - {self.n_samples} samples, ")
        for key in errors.keys():
            errors[key] /= self.n_samples
            print(f"{key}: {errors[key]:.4f}", end=", ")
        print()

        # on last batch, log model outputs
        if self.log_output and self.wandb_log:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)

        return errors

    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss, epoch):
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch
        epoch: int
            current epoch
        Returns
        -------
        loss: float | Tensor
            float value of training loss
        gt: torch.Tensor
            ground truth
        out: torch.Tensor
            model output
        """

        self.optimizer.zero_grad(set_to_none=True)
        ## ADDED FOR WOS
        if epoch == 0:
            p_arr = sample["wos_estimate"]
            self.cache[idx] = p_arr.detach().cpu()
        else:
            p_arr = sample["wos_estimate"]
            self.cache[idx] = (
                10 * p_arr.detach().cpu() + (epoch * 10) * self.cache[idx]
            ) / ((1 + epoch) * 10)
            p_arr = self.cache[idx]

        if self.regularizer:
            self.regularizer.reset()

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)
            }

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1
        start = time.time()
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)

        assert (
            torch.isnan(out).any() == False
        ), f"NaN detected in model output at epoch {epoch}, batch {idx}"
        # ADDED FOR WOS
        # grad_loss_fn = torch.nn.MSELoss()
        gt = sample["y"].detach().clone()
        sample["y"] = p_arr.to(self.device)
        del p_arr

        if (
            self.epoch == 0
            and idx == 0
            and self.verbose
            and isinstance(out, torch.Tensor)
        ):
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss += training_loss(out, **sample)
        else:
            loss += training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss
        return loss, gt, out, sample

    def eval_one_batch(
        self, sample: dict, eval_losses: dict, return_output: bool = False
    ):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        p_arr = sample["wos_estimate"]
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)
            }
        gt = sample["y"].to(self.device)
        sample["y"] = p_arr.to(self.device)
        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)
        mse = torch.nn.MSELoss()
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss

        err_gt = mse(out, gt).item()
        err_wos = mse(out, sample["y"]).item()

        eval_step_losses["avg_loss_wrt_gt"] = err_gt
        eval_step_losses["avg_wos_loss_wrt_gt"] = err_wos

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def eval_one_batch_autoreg(
        self,
        sample: dict,
        eval_losses: dict,
        return_output: bool = False,
        max_steps: int = None,
    ):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        max_steps: int
            number of timesteps to roll out
            typically the full trajectory length
            If max_steps is none, runs until the full length

            .. note::
                If a value for ``max_steps`` is not provided, a data_processor
                must be provided to handle rollout logic.
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs


        """
        eval_step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}
        # eval_rollout_losses = {loss_name: 0. for loss_name in eval_losses.keys()}

        t = 0
        if max_steps is None:
            max_steps = float("inf")

        # only increment the sample count once
        sample_count_incr = False

        while sample is not None and t < max_steps:

            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                # load data to device if no preprocessor exists
                sample = {
                    k: v.to(self.device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }

            if sample is None:
                break

            # only increment the sample count once
            if not sample_count_incr:
                self.n_samples += sample["y"].shape[0]
                sample_count_incr = True

            out = self.model(**sample)

            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)

            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, **sample)
                eval_step_losses[loss_name] += step_loss

            t += 1
        # average over all steps of the final rollout
        for loss_name in eval_step_losses.keys():
            eval_step_losses[loss_name] /= t

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: float = None,
        lr: float = None,
    ):
        """Basic method to log results
        from a single training epoch.


        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def log_eval(self, epoch: int, eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader

        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value

        msg = f"Eval: " + msg[:-2]  # cut off last comma+space
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`

        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError(
                "Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt."
            )
        # returns model, loads other modules if provided
        self.model, self.optimizer, self.scheduler, self.regularizer, resume_epoch = (
            load_training_state(
                save_dir=save_dir,
                save_name=save_name,
                model=self.model,
                optimizer=self.optimizer,
                regularizer=self.regularizer,
                scheduler=self.scheduler,
            )
        )

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer resuming from epoch {resume_epoch}")

    def checkpoint(self, save_dir):
        """checkpoint saves current training state
        to a directory for resuming later. Only saves
        training state on the first GPU.
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if comm.get_local_rank() == 0:
            if self.save_best is not None:
                save_name = "best_model"
            else:
                save_name = "model"
            save_training_state(
                save_dir=save_dir,
                save_name=save_name,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                regularizer=self.regularizer,
                epoch=self.epoch,
            )
            if self.verbose:
                print(f"[Rank 0]: saved training state to {save_dir}")
