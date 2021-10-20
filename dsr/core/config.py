import os
import time
import yaml
import shutil
import argparse
import operator
import itertools
from os.path import join
from loguru import logger
from functools import reduce
from yacs.config import CfgNode as CN
from typing import Dict, List, Union, Any
from flatten_dict import flatten, unflatten

##### CONSTANTS #####
DATASET_NPZ_PATH = 'dsr_data/dataset_extras'

H36M_ROOT = 'dsr_data/dataset_folders/h36m'
COCO_ROOT = 'dsr_data/dataset_folders/coco'
MPI_INF_3DHP_ROOT = 'dsr_data/dataset_folders/mpi_inf_3dhp'
PW3D_ROOT = 'dsr_data/dataset_folders/3dpw'

JOINT_REGRESSOR_TRAIN_EXTRA = 'dsr_data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'dsr_data/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = 'dsr_data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'dsr_data/smpl'

OPENPOSE_PATH = 'datasets/openpose'

DATASET_FOLDERS = {
    'h36m': H36M_ROOT,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
    'coco': COCO_ROOT,
    '3dpw': PW3D_ROOT,
}

DATASET_FILES = [
    {
        'h36m-p1': 'h36m_valid_protocol1.npz',
        'h36m-p2': 'h36m_valid_protocol2.npz',
        'mpi-inf-3dhp': 'mpi_inf_3dhp_valid.npz',
        '3dpw': '3dpw_test_with_mmpose.npz',
    },
    {
        'h36m': 'h36m_train.npz',
        'coco': 'coco_2014_train.npz',
        'mpi-inf-3dhp': 'mpi_inf_3dhp_train.npz',
        '3dpw': '3dpw_train.npz',
    }
]

##### CONFIGS #####
hparams = CN()

# General settings
hparams.LOG_DIR = 'logs/experiments'
hparams.METHOD = 'spin' # spin/dsr
hparams.EXP_NAME = 'default'
hparams.EXP_ID = ''
hparams.RUN_TEST = False
hparams.SEED_VALUE = -1
hparams.PL_LOGGING = True

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.NOISE_FACTOR = 0.4
hparams.DATASET.ROT_FACTOR = 30
hparams.DATASET.SCALE_FACTOR = 0.25
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.SHUFFLE_VAL = True
hparams.DATASET.TRAIN_DS = 'h36m'
hparams.DATASET.VAL_DS = '3dpw'
hparams.DATASET.NUM_IMAGES = -1
hparams.DATASET.IMG_RES = 224
hparams.DATASET.FOCAL_LENGTH = 5000.
hparams.DATASET.IGNORE_3D = False
hparams.DATASET.ONLY_IUV = False
hparams.DATASET.MESH_COLOR = 'light_pink'
hparams.DATASET.GENDER_EVAL = True
hparams.DATASET.TRAIN_3DPW = False

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 0.0001
hparams.OPTIMIZER.WD = 0.0
hparams.OPTIMIZER.MM = 0.9

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED = None
hparams.TRAINING.PRETRAINED_LIT = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 40
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True
hparams.TRAINING.SAVE_IMAGES = False
hparams.TRAINING.USE_AUGM = True

# Training process hparams
hparams.TESTING = CN()
hparams.TESTING.SAVE_IMAGES = False
hparams.TESTING.SAVE_RESULTS = False
hparams.TESTING.SIDEVIEW = True
hparams.TESTING.LOG_FREQ_TB_IMAGES = 50
hparams.TESTING.DISP_ALL = True

# SPIN method hparams
hparams.SPIN = CN()
hparams.SPIN.BACKBONE = 'resnet50'

hparams.SPIN.SHAPE_LOSS_WEIGHT = 0
hparams.SPIN.KEYPOINT_LOSS_WEIGHT = 5.
hparams.SPIN.KEYPOINT_NATIVE_LOSS_WEIGHT = 5.
hparams.SPIN.POSE_LOSS_WEIGHT = 1.
hparams.SPIN.BETA_LOSS_WEIGHT = 0.001
hparams.SPIN.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.SPIN.GT_TRAIN_WEIGHT = 1.
hparams.SPIN.LOSS_WEIGHT = 60.

# DSR method hparams
hparams.DSR = CN()
hparams.DSR.BACKBONE = 'resnet50'
hparams.DSR.SHAPE_LOSS_WEIGHT = 0
hparams.DSR.KEYPOINT_LOSS_WEIGHT = 5.
hparams.DSR.KEYPOINT_NATIVE_LOSS_WEIGHT = 5.
hparams.DSR.POSE_LOSS_WEIGHT = 1.
hparams.DSR.BETA_LOSS_WEIGHT = 0.001
hparams.DSR.DSR_MC_LOSS_WEIGHT = 0.2
hparams.DSR.DSR_C_LOSS_WEIGHT = 0.2
hparams.DSR.OPENPOSE_TRAIN_WEIGHT = 0.
hparams.DSR.GT_TRAIN_WEIGHT = 1.
hparams.DSR.LOSS_WEIGHT = 60.
hparams.DSR.GAMMA_VAL = 1.0e-1
hparams.DSR.SIGMA_VAL = 1.0e-7
hparams.DSR.DSR_MC_LOSS_TYPE = 'DistM'
hparams.DSR.START_DSR = -1

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()


def get_grid_search_configs(config, excluded_keys=[]):
    """
    :param config: dictionary with the configurations
    :return: The different configurations
    """

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []

    for k,v in flattened_config_dict.items():
        if isinstance(v,list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool) :
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v,list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ['True', 'False']):
                exp[param_name] = [True if x == 'True' else False for x in param_value]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False


        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params


def run_grid_search_experiments(
        cfg_id,
        cfg_file,
        script='main.py',
):
    cfg = yaml.load(open(cfg_file))

    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    logger.info(f'Grid search hparams: \n {hyperparams}')

    different_configs = [update_hparams_from_dict(c) for c in different_configs]
    logger.info(f'======> Number of experiment configurations is {len(different_configs)}')

    config_to_run = CN(different_configs[cfg_id])

    # ==== create logdir using hyperparam settings
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{config_to_run.EXP_NAME}'

    def get_from_dict(dict, keys):
        return reduce(operator.getitem, keys, dict)

    exp_id = ''
    for hp in hyperparams:
        v = get_from_dict(different_configs[cfg_id], hp.split('/'))
        exp_id += f'{hp.replace("/", ".").replace("_", "").lower()}-{v}'


    config_to_run.EXP_ID = f'{config_to_run.EXP_NAME}'
    if exp_id:
        logdir += f'_{exp_id}'
        config_to_run.EXP_ID += f'/{exp_id}'

    logdir = os.path.join(config_to_run.LOG_DIR, config_to_run.METHOD, config_to_run.EXP_NAME, logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=os.path.join(config_to_run.LOG_DIR, 'config.yaml'))

    config_to_run.LOG_DIR = logdir

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)),
        os.path.join(config_to_run.LOG_DIR, 'config_to_run.yaml')
    )

    return config_to_run
