import os
import torch
import numpy as np
from loguru import logger
import pytorch_lightning as pl

from ..core.config import SMPL_MEAN_PARAMS

def prepare_statedict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if 'model' in name:
            name = name.replace('model.', '')
        if 'backbone' in name:
            name = name.replace('backbone.', '')
        if 'head' in name:
            name = name.replace('head.', '')
        new_state_dict[name] = param
    return new_state_dict

def add_smpl_params_to_dict(state_dict):
    mean_params = np.load(SMPL_MEAN_PARAMS)
    init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
    init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
    init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
    state_dict['init_pose'] = init_pose
    state_dict['init_shape'] = init_shape
    state_dict['init_cam'] = init_cam
    return state_dict
    
def set_seed(seed_value):
    if seed_value >= 0:
        logger.info(f'Seed value for the experiment {seed_value}')
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        pl.trainer.seed_everything(seed_value)

def load_pretrained_model(model, pt_file, strict=False, overwrite_shape_mismatch=True):

    state_dict = torch.load(pt_file)['state_dict']
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        logger.warning(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                       f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')
                        del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
