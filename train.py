import os
import torch
import pprint
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dsr.core.lightning import LitModule
from dsr.utils.os_utils import copy_code
from dsr.utils.train_utils import set_seed
from dsr.core.config import run_grid_search_experiments

def main(hparams):

    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    copy_code(
        output_folder=log_dir,
        curr_folder=os.path.dirname(os.path.abspath(__file__))
    )

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')
    

    model = LitModule(hparams=hparams).to(device)

    ckpt_callback = False

    # Turn on PL logging and Checkpoint saving
    tb_logger = None
    if hparams.PL_LOGGING == True:
        ckpt_callback = ModelCheckpoint(
            monitor='val_loss',
            verbose=True,
            save_top_k=5,
            mode='min',
        )
        # initialize tensorboard logger
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name='tb_logs',
        )

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        gpus=1,
        logger=tb_logger,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        log_save_interval=hparams.TRAINING.LOG_SAVE_INTERVAL,
        terminate_on_nan=True,
        default_root_dir=log_dir,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        checkpoint_callback=ckpt_callback,
        reload_dataloaders_every_epoch=hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
        log_gpu_memory=True,
    )

    if hparams.RUN_TEST:
        logger.info('*** Started testing ***')
        trainer.test(model=model)
    else:
        logger.info('*** Started training ***')
        trainer.fit(model)
        trainer.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        script='train.py',
    )

    main(hparams)
