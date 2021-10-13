import os
import cv2
import json
import math
import torch
import joblib
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from smplx import SMPL as SMPL_native
from torch.utils.data import DataLoader

from . import config
from . import constants
from ..models import SMPL
from ..utils.renderer import Renderer
from ..utils.eval_utils import reconstruction_error, compute_error_verts
from ..utils.dataloader import CheckpointDataLoader
from ..dataset import BaseDataset, MixedDataset
from ..utils.geometry import estimate_translation, perspective_projection, rotation_matrix_to_angle_axis


class LitModule(pl.LightningModule):

    def __init__(self, hparams):
        super(LitModule, self).__init__()

        self.hparams = hparams

        if self.hparams.METHOD == 'dsr':
            from ..models import HMR
            from ..losses import DSRLoss
            self.model = HMR(
                backbone=self.hparams.DSR.BACKBONE,
                img_res=self.hparams.DATASET.IMG_RES,
                pretrained=self.hparams.TRAINING.PRETRAINED,
            )
            self.loss_fn = DSRLoss(
                shape_loss_weight=self.hparams.DSR.SHAPE_LOSS_WEIGHT,
                keypoint_loss_weight=self.hparams.DSR.KEYPOINT_LOSS_WEIGHT,
                pose_loss_weight=self.hparams.DSR.POSE_LOSS_WEIGHT,
                beta_loss_weight=self.hparams.DSR.BETA_LOSS_WEIGHT,
                openpose_train_weight=self.hparams.DSR.OPENPOSE_TRAIN_WEIGHT,
                srp_loss_weight=self.hparams.DSR.SRP_LOSS_WEIGHT,
                srv_loss_weight=self.hparams.DSR.SRV_LOSS_WEIGHT,
                gt_train_weight=self.hparams.DSR.GT_TRAIN_WEIGHT,
                loss_weight=self.hparams.DSR.LOSS_WEIGHT,
                gamma_val=self.hparams.DSR.GAMMA_VAL,
                sigma_val=self.hparams.DSR.SIGMA_VAL,
                baseline=self.hparams.DSR.BASELINE,
                srp_loss_type=self.hparams.DSR.SRP_LOSS_TYPE,
            )
        elif self.hparams.METHOD == 'spin' or self.hparams.DATASET.ONLY_IUV == True:
            from ..models import HMR
            from ..losses import HMRLoss
            self.model = HMR(
                backbone=self.hparams.SPIN.BACKBONE,
                img_res=self.hparams.DATASET.IMG_RES,
                pretrained=self.hparams.TRAINING.PRETRAINED,
            )
            self.loss_fn = HMRLoss(
                shape_loss_weight=self.hparams.SPIN.SHAPE_LOSS_WEIGHT,
                keypoint_loss_weight=self.hparams.SPIN.KEYPOINT_LOSS_WEIGHT,
                pose_loss_weight=self.hparams.SPIN.POSE_LOSS_WEIGHT,
                beta_loss_weight=self.hparams.SPIN.BETA_LOSS_WEIGHT,
                openpose_train_weight=self.hparams.SPIN.OPENPOSE_TRAIN_WEIGHT,
                gt_train_weight=self.hparams.SPIN.GT_TRAIN_WEIGHT,
                loss_weight=self.hparams.SPIN.LOSS_WEIGHT,
            )
        else:
            logger.error(f'{self.hparams.METHOD} is undefined!')
            exit()

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )

        self.smpl_native = SMPL_native(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smpl.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        # Initialiatize variables related to evaluation
        self.best_result = math.inf
        self.best_pampjpe = math.inf
        self.best_mpjpe = math.inf
        self.best_v2v = math.inf
        self.val_accuracy_results = []
        self.init_evaluation_variables()

        self.start_dsr = True
        self.pl_logging = self.hparams.PL_LOGGING

    def init_evaluation_variables(self):
        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = [] # np.zeros(len(self.val_ds))
        self.val_pampjpe = [] # np.zeros(len(self.val_ds))
        self.val_v2v = []

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'mpjpe': [], # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [], # np.zeros((len(self.val_ds), 14)),
            'v2v': [],
        }

        # use this to save the errors for each image
        if self.hparams.TESTING.SAVE_IMAGES:
            self.val_images_errors = []

        if self.hparams.TESTING.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam_t'] = []


    def forward(self):
        return None

    def training_step(self, batch, batch_nb):

        batch_size = None
        # Get data from the batch
        inputs = batch['img']
        batch_size = inputs.shape[0]

        gt_keypoints_2d = batch['keypoints']  # 2D keypoints
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters
        gt_joints = batch['pose_3d']  # 3D pose
        has_smpl = batch['has_smpl'].bool()  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = batch['has_pose_3d'].bool()  # flag that indicates whether 3D pose is valid
        is_flipped = batch['is_flipped'].bool()  # flag that indicates whether image was flipped during data augmentation
        rot_angle = batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = batch['dataset_name']  # name of the dataset the image comes from
        indices = batch['sample_index']  # index of example inside its dataset

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = \
            0.5 * self.hparams.DATASET.IMG_RES * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_size=self.hparams.DATASET.IMG_RES,
        )

        opt_joints = gt_model_joints
        opt_pose = batch['pose']
        opt_betas = batch['betas']
        opt_cam_t = gt_cam_t

        batch['gt_cam_t'] = gt_cam_t
        batch['vertices'] = gt_vertices

        camera_center = torch.zeros(batch_size, 2, device=self.device)
        opt_keypoints_2d = perspective_projection(
            opt_joints,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=opt_cam_t,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            camera_center=camera_center,
        )

        opt_keypoints_2d = opt_keypoints_2d / (self.hparams.DATASET.IMG_RES / 2.)

        opt_native_model_joints = self.smpl_native(
            betas=opt_betas,
            body_pose=opt_pose[:, 3:],
            global_orient=opt_pose[:, :3]
        ).joints[:, :24, :]

        opt_smpl_keypoints_2d = perspective_projection(
            opt_native_model_joints,
            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=opt_cam_t,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            camera_center=camera_center,
        )
        # Normalize keypoints to [-1,1]
        opt_smpl_keypoints_2d = opt_smpl_keypoints_2d / (self.hparams.DATASET.IMG_RES / 2.)
        batch['smpl_keypoints'] = opt_smpl_keypoints_2d

        # Forward pass
        pred = self.model(inputs)

        if self.hparams.DSR.START_DSR > -1:
            self.start_dsr = (self.current_epoch > self.hparams.DSR.START_DSR)

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch, start_dsr=self.start_dsr)
        tensorboard_logs = loss_dict

        if batch_nb % self.hparams.TRAINING.LOG_FREQ_TB_IMAGES == 0:
            self.train_summaries(input_batch=batch, output=pred)

        return {'loss': loss, 'log': tensorboard_logs}

    def train_summaries(self, input_batch, output):

        if self.pl_logging == False and self.hparams.TRAINING.SAVE_IMAGES == False:
            return

        if self.pl_logging == True:
            tb_logger = self.logger[0]
            comet_logger = self.logger[1]

        images = input_batch['img']
        iuvs = None

        pred_vertices = output['smpl_vertices'].detach()
        opt_vertices = input_batch['vertices']

        pred_cam_t = output['pred_cam_t'].detach()
        opt_cam_t = input_batch['gt_cam_t']

        pred_kp_2d = output['pred_kp2d'].detach() if 'pred_kp2d' in output.keys() else None
        gt_kp_2d = input_batch['smpl_keypoints']

        grphs = input_batch['grph'] if 'grph' in input_batch.keys() else None
        sr_rend = output['sr_rend'].detach() if 'sr_rend' in output.keys() else None

        images_pred = self.renderer.visualize_tb(
            vertices=pred_vertices,
            camera_translation=pred_cam_t,
            images=images,
            kp_2d=gt_kp_2d,
            sideview=self.hparams.TESTING.SIDEVIEW,
        )
        images_opt = self.renderer.visualize_tb(
            vertices=opt_vertices,
            camera_translation=opt_cam_t,
            images=images,
            kp_2d=gt_kp_2d,
            sideview=self.hparams.TESTING.SIDEVIEW,
        )

        if self.pl_logging == True:
            tb_logger.experiment.add_image('pred_shape', images_pred, self.global_step)
            tb_logger.experiment.add_image('opt_shape', images_opt, self.global_step)

        if self.hparams.TRAINING.SAVE_IMAGES == True:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
            save_dir = os.path.join(self.hparams.LOG_DIR, 'train_output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                    os.path.join(save_dir, f'result_{self.global_step:05d}.png'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
            )

    def validation_step(self, batch, batch_nb):

        curr_batch_size = None
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        # Get data from the batch
        inputs = batch['img']
        curr_batch_size = inputs.shape[0]

        with torch.no_grad():
            pred = self.model(inputs)
            pred_vertices = pred['smpl_vertices']

        joint_mapper_h36m = constants.H36M_TO_J17 if self.val_ds.dataset == 'mpi-inf-3dhp' \
            else constants.H36M_TO_J14

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

        # For 3DPW get the 14 common joints from the rendered shape
        gt_keypoints_3d = batch['pose_3d'].cuda()

        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        # Reconstuction_error
        r_error, r_error_per_joint = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None,
        )
        
        # Per-vertex error
        if 'vertices' in batch.keys():
            gt_vertices = batch['vertices'].cuda()

            v2v = compute_error_verts(
                pred_verts=pred_vertices.cpu().numpy(),
                target_verts=gt_vertices.cpu().numpy(),
            )
            self.val_v2v += v2v.tolist()
        else:
            v2v = np.zeros_like(error)
            self.val_v2v += np.zeros_like(error).tolist()
            
        self.val_mpjpe += error.tolist()
        self.val_pampjpe += r_error.tolist()

        error_per_joint = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()

        self.evaluation_results['mpjpe'] += error_per_joint[:,:14].tolist()
        self.evaluation_results['pampjpe'] += r_error_per_joint[:,:14].tolist()
        self.evaluation_results['v2v'] += v2v.tolist()
        self.evaluation_results['imgname'] += imgnames
        self.evaluation_results['dataset_name'] += dataset_names

        if self.hparams.TESTING.SAVE_RESULTS: 
            tolist = lambda x: [i for i in x.cpu().numpy()]
            self.evaluation_results['pose'] += tolist(pred['pred_pose'])
            self.evaluation_results['shape'] += tolist(pred['pred_shape'])
            self.evaluation_results['cam_t'] += tolist(pred['pred_cam_t'])

        if batch_nb % self.hparams.TESTING.LOG_FREQ_TB_IMAGES == 0:
            self.validation_summaries(batch, pred, batch_nb, error, r_error)

        tensorboard_logs = {
            'val/val_mpjpe_step': error.mean(),
            'val/val_pampjpe_step': r_error.mean(),
            'val/val_v2v_step': v2v.mean(),
        }

        return {'val_loss': r_error.mean(), 'log': tensorboard_logs}

    def validation_summaries(self, input_batch, output, batch_idx, error=None, r_error=None):

        if self.pl_logging == False and self.hparams.TESTING.SAVE_IMAGES == False:
            return

        if self.pl_logging == True:
            tb_logger = self.logger[0]
            comet_logger = self.logger[1]

        iuvs = None
        images = input_batch['img']

        pred_vertices = output['smpl_vertices'].detach()
        pred_cam_t = output['pred_cam_t'].detach()
        pred_kp_2d = output['pred_kp2d'].detach() if 'pred_kp2d' in output.keys() else None

        images_pred = self.renderer.visualize_tb(
            vertices=pred_vertices,
            camera_translation=pred_cam_t,
            images=images,
            kp_2d=pred_kp_2d,
            nb_max_img=4,
            sideview=self.hparams.TESTING.SIDEVIEW,
        )

        if self.pl_logging == True:
            tb_logger.experiment.add_image('val_pred_shape', images_pred, self.global_step)

        # tb_logger.experiment.add_image('pred_shape', images_pred, self.global_step)

        if self.hparams.TESTING.SAVE_IMAGES == True:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
            save_dir = os.path.join(self.hparams.LOG_DIR, 'val_output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                    os.path.join(save_dir, f'result_{self.global_step:05d}_{batch_idx}.png'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
            )

    def validation_epoch_end(self, outputs):

        if self.pl_logging == True:
            tb_logger = self.logger[0]
            comet_logger = self.logger[1]

        self.val_mpjpe = np.array(self.val_mpjpe)
        self.val_pampjpe = np.array(self.val_pampjpe)
        self.val_v2v = np.array(self.val_v2v)

        for k,v in self.evaluation_results.items():
            self.evaluation_results[k] = np.array(v)

        self.evaluation_results['epoch'] = self.current_epoch

        avg_mpjpe, avg_pampjpe = 1000 * self.val_mpjpe.mean(), 1000 * self.val_pampjpe.mean()
        avg_v2v = 1000 * self.val_v2v.mean()

        logger.info(f'***** Epoch {self.current_epoch} *****')
        logger.info('MPJPE: ' + str(avg_mpjpe))
        logger.info('PA-MPJPE: ' + str(avg_pampjpe))
        logger.info('V2V (mm): ' + str(avg_v2v))

        avg_mpjpe, avg_pampjpe, avg_v2v = torch.tensor(avg_mpjpe), torch.tensor(avg_pampjpe), torch.tensor(avg_v2v)

        acc = {
            'val_mpjpe': avg_mpjpe.item(),
            'val_pampjpe': avg_pampjpe.item(),
            'val_v2v': avg_v2v.item(),
        }
        self.val_save_best_results(acc)

        # Best model selection criterion - 1.5 * PAMPJPE + MPJPE
        best_result = 1.5 * avg_pampjpe.clone().cpu().numpy() + avg_mpjpe.clone().cpu().numpy()
        if best_result < self.best_result:
            logger.info(f'Best Model Criteria Met: Current Score -> {best_result} \
                        | Previous Score -> {self.best_result}')
            self.best_result = best_result
            self.best_pampjpe = avg_pampjpe
            self.best_mpjpe = avg_mpjpe
            self.best_v2v = avg_v2v
            joblib.dump(
                self.evaluation_results,
                os.path.join(self.hparams.LOG_DIR, 
                    f'evaluation_results_{self.hparams.DATASET.VAL_DS}.pkl')
            )

        tensorboard_logs = {
            'val/val_mpjpe': avg_mpjpe,
            'val/val_pampjpe': avg_pampjpe,
            'val/val_v2v': avg_v2v,
            'val/best_pampjpe': self.best_pampjpe,
            'val/best_mpjpe': self.best_mpjpe,
            'val/best_v2v': self.best_v2v,
            'step': self.current_epoch,
        }

        self.init_evaluation_variables()

        return {'val_loss': avg_pampjpe, 'val_mpjpe': avg_mpjpe, 'val_pampjpe': avg_pampjpe, \
                'val_v2v': avg_v2v, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def train_dataset(self):
        train_ds = None
        if self.hparams.METHOD == 'dsr':
            self.hparams.DATASET.BASELINE = self.hparams.DSR.BASELINE
            self.hparams.DATASET.SRP_PROB = self.hparams.DSR.SRP_PROB
            self.hparams.DATASET.USE_CLASS_WEIGHT = self.hparams.DSR.USE_CLASS_WEIGHT

        if self.hparams.DATASET.TRAIN_DS == 'all':
            train_ds = MixedDataset(
                self.hparams.DATASET,
                self.hparams.METHOD,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                num_images=self.hparams.DATASET.NUM_IMAGES,
                is_train=True
            )
        elif self.hparams.DATASET.TRAIN_DS in config.DATASET_FOLDERS.keys():
            train_ds = BaseDataset(
                self.hparams.DATASET,
                self.hparams.METHOD,
                use_augmentation=self.hparams.TRAINING.USE_AUGM,
                dataset=self.hparams.DATASET.TRAIN_DS,
                num_images=self.hparams.DATASET.NUM_IMAGES,
            )
        else:
            logger.error(f'{self.hparams.DATASET.TRAIN_DS} is undefined!')
            exit()
        return train_ds

    def validation_dataset(self):
        if self.hparams.METHOD == 'dsr':
            self.hparams.DATASET.BASELINE = self.hparams.DSR.BASELINE
            self.hparams.DATASET.SRP_PROB = self.hparams.DSR.SRP_PROB
            self.hparams.DATASET.USE_CLASS_WEIGHT = self.hparams.DSR.USE_CLASS_WEIGHT

        val_ds = BaseDataset(
            self.hparams.DATASET,
            self.hparams.METHOD,
            dataset=self.hparams.DATASET.VAL_DS,
            num_images=self.hparams.DATASET.NUM_IMAGES,
            is_train=False,
        )
        return val_ds

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return CheckpointDataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )

    def val_dataloader(self):
        self.val_ds = self.validation_dataset()
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            shuffle=False,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def val_save_best_results(self, acc):
        json_file = os.path.join(self.hparams.LOG_DIR, 'val_accuracy_results.json')
        self.val_accuracy_results.append([self.global_step, acc])
        with open(json_file, 'w') as f:
            json.dump(self.val_accuracy_results, f, indent=4)
