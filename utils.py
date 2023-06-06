import torch
import os
import logging
from omegaconf import OmegaConf


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device, ddp=True):
  if not os.path.exists(ckpt_dir):
    makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    if ddp:
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
    else:
        state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    if state['scaler'] is not None:
        state['scaler'].load_state_dict(loaded_state['scaler'])
    return state


def load_denoising_model(ckpt_dir, model, device=torch.device('cpu')):
    if not os.path.exists(ckpt_dir):
        raise ValueError(f"No checkpoint found at {ckpt_dir}.")
    loaded_state = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(loaded_state['model'], strict=False)
    return model


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].module.state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'scaler': state['scaler'].state_dict() if state['scaler'] else None
  }
  torch.save(saved_state, ckpt_dir)

