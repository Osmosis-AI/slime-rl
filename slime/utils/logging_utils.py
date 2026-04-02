import logging

from . import wandb_utils
from .tracking import TrackingManager

_LOGGER_CONFIGURED = False
_manager = TrackingManager()


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def update_tracking_open_metrics(args, router_addr):
    if not getattr(args, "use_wandb", False):
        return
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step)


def log_samples(samples, step=None):
    _manager.log_samples(samples, step=step)


def log_checkpoint(checkpoint_dir, metadata=None):
    _manager.log_checkpoint(checkpoint_dir, metadata=metadata)


def log_model_params(total_params, trainable_params):
    _manager.log_model_params(total_params, trainable_params)


def finish_tracking(args=None):
    _manager.finish()
