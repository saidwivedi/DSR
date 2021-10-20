import torch
import numpy as np
import torch.nn as nn
import soft_renderer as sr

from loguru import logger
from ..models import SMPL
from ..core import constants
from ..core.config import SMPL_MODEL_DIR
from ..utils.geometry import batch_rodrigues

from ..semantic_rendering.data_utils import convert_valid_labels, convert_camT_to_proj_mat
from ..semantic_rendering.loss import dsr_mc_loss

class HMRLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint_loss_weight=5.,
            pose_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            loss_weight=60.,
    ):
        super(HMRLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict

class DSRLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint_loss_weight=5.,
            pose_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            dsr_mc_loss_weight=0.2,
            dsr_c_loss_weight=0.2,
            gt_train_weight=1.,
            loss_weight=60.,
            gamma_val=1.0e-1,
            sigma_val=1.0e-7,
            baseline=False,
            dsr_mc_loss_type='DistM',
    ):
        super(DSRLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight

        self.gamma_val = gamma_val
        self.sigma_val = sigma_val
        self.background_color = [0.0, 0.0, 0.0]
        self.light_intensity_ambient = 1.0
        self.light_intensity_directionals = 0
        self.img_size = constants.IMG_RES
        self.criterion_dsr_c = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.baseline = baseline
        self.dsr_mc_loss_type = 'nIOU' if self.baseline else dsr_mc_loss_type
        smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
        self.smpl_faces = torch.from_numpy(smpl.faces.astype('int32'))[None].cuda()

        self.dsr_mc_loss_weight = dsr_mc_loss_weight
        self.dsr_c_loss_weight = dsr_c_loss_weight

    def sr_losses(self, gt_batch, vertices, textures, cam_t, dsr_mc_dist_mat, dsr_c_img_label, \
                  dsr_mc_img_label, valid_labels_dsr_mc, valid_labels_dsr_c, dsr_c_class_weight, start_dsr):

        def de_norm(images):
            images = images * torch.tensor([0.229, 0.224, 0.225], \
                                            device=images.device).reshape(1,3,1,1)
            images = images + torch.tensor([0.485, 0.456, 0.406], \
                                            device=images.device).reshape(1,3,1,1)
            return images

        def debug_rend_out(imgs, grphs, rend_out, dsr_mc_image_label, dsr_mc_dist_mat, idx):
            disp_img_list = []
            # Image
            disp_img = de_norm(imgs[idx])[0]
            disp_img = disp_img.detach().clone().cpu().numpy().transpose((1,2,0))
            disp_img = (disp_img * 255).astype(np.uint8)
            disp_img_list.append(disp_img)
            # Graphonomy
            disp_img = grphs[idx].detach().clone().cpu().numpy().transpose((1,2,0))
            disp_img = (disp_img * 255).astype(np.uint8)
            disp_img_list.append(disp_img)
            # DSR_MC Dist Mat
            disp_img = dsr_mc_dist_mat.detach().clone().cpu().numpy()[0].transpose((1, 2, 0))
            disp_img *= (255.0/disp_img.max())
            disp_img = disp_img.astype(np.uint8)
            disp_img_list.append(disp_img)
            # DSR_MC Image Label
            disp_img = dsr_mc_image_label.detach().clone().cpu().numpy().transpose((1, 2, 0))
            disp_img = (disp_img * 255).astype(np.uint8)
            disp_img_list.append(disp_img)
            #SR-Pixel - Prob
            disp_img = rend_out[0,:3].detach().clone().cpu().numpy().transpose((1, 2, 0))
            disp_img = (disp_img * 255).astype(np.uint8)
            disp_img_list.append(disp_img)
            #SR-Pixel - Silhouette
            disp_img = rend_out[0,3].detach().repeat(3,1,1).cpu().numpy().transpose((1, 2, 0))
            disp_img = (disp_img * 255).astype(np.uint8)
            disp_img_list.append(disp_img)
            #SR-Vertex
            if not self.baseline:
                for i in range(4):
                    disp_img = rend_out[i+1,:3,:,:].detach().cpu().numpy().transpose(1,2,0)
                    disp_img_list.append(disp_img)
            show_imgs(disp_img_list, save_img=True, filename=f'dump_dir/dsr_srloss/rend_{idx}.png')

        batch_size = vertices.shape[0]
        rend_dim = textures.shape[1]

        loss_dsr_mc = torch.zeros(batch_size, device=vertices.device)
        loss_dsr_c = torch.zeros(batch_size, device=vertices.device)

        # Don't compute SR losses if all loss_weights are zero
        if self.dsr_mc_loss_weight == 0 and self.dsr_c_loss_weight == 0:
            #logger.warning(f'SR losses turned off')
            return loss_dsr_mc.mean(), loss_dsr_c.mean()

        # Late start of SR Losses
        if not start_dsr:
            #logger.warning(f'dsr loss has not started')
            return loss_dsr_mc.mean(), loss_dsr_c.mean()

        # Prepare data for rendering the entire batch together
        dsr_mc_dist_mat = dsr_mc_dist_mat.permute(0,3,1,2)
        dsr_mc_img_label = dsr_mc_img_label.permute(0,3,1,2)
        textures = textures.unsqueeze(3)
        dsr_c_img_label = dsr_c_img_label.long()
        P = convert_camT_to_proj_mat(cam_t.cpu()).cuda() # TODO: check why sending directly cuda doesn't work

        batch_smpl_faces = self.smpl_faces.expand(rend_dim*batch_size, self.smpl_faces.shape[1], self.smpl_faces.shape[2])
        batch_vertices = torch.repeat_interleave(vertices, repeats=rend_dim, dim=0)
        batch_P = torch.repeat_interleave(P, repeats=rend_dim, dim=0)
        batch_textures = textures.view(rend_dim*batch_size, textures.shape[2], textures.shape[3], textures.shape[4])

        renderer = sr.SoftRenderer(P=batch_P,
                   camera_mode='projection',
                   gamma_val=self.gamma_val,
                   sigma_val=self.sigma_val,
                   orig_size=self.img_size,
                   image_size=self.img_size,
                   background_color=self.background_color,
                   light_intensity_ambient=self.light_intensity_ambient,
                   light_intensity_directionals=self.light_intensity_directionals)
        rend_out = renderer(batch_vertices, batch_smpl_faces, batch_textures)


        # Calculate loss for individual sample in batch to avoid computing loss
        # for samples without graphonomy labels
        for idx in range(batch_size):

            if len(valid_labels_dsr_mc[idx]) == 0:
                #logger.warning(f'No valid labels')
                continue

            cur_dsr_mc_dist_mat = dsr_mc_dist_mat[None,idx]
            cur_dsr_c_img_label = dsr_c_img_label[None,idx]
            cur_dsr_mc_img_label = dsr_mc_img_label[idx]
            cur_dsr_c_class_weight = dsr_c_class_weight[idx]

            start_index = rend_dim * idx
            cur_rend_out = rend_out[start_index:start_index+rend_dim]

            # SR-Pixel
            rend_dsr_mc = cur_rend_out[0]
            loss_dsr_mc[idx] = dsr_mc_loss(rend_dsr_mc, cur_dsr_mc_img_label, cur_dsr_mc_dist_mat, \
                                     self.dsr_mc_loss_type, self.baseline)

            # SR-Vertex
            if rend_dim > 1:
                self.criterion_dsr_c.weight = cur_dsr_c_class_weight
                rend_dsr_c = cur_rend_out[1:,:3].mean(1).unsqueeze(0)
                loss_dsr_c[idx] = self.criterion_dsr_c(rend_dsr_c, cur_dsr_c_img_label)

            if torch.isnan(loss_dsr_c[idx]) or torch.isnan(loss_dsr_mc[idx]) or \
               torch.isinf(loss_dsr_c[idx]) or torch.isinf(loss_dsr_mc[idx]):
                imgs, imgname, grphs = gt_batch['img'], gt_batch['imgname'], gt_batch['grph']
                debug_rend_out(imgs, grphs, cur_rend_out, cur_dsr_mc_img_label, \
                               cur_dsr_mc_dist_mat, idx)
                logger.warning(f'loss is nan for {imgname[idx]}')
                logger.warning(f'current_rend - {torch.unique(cur_rend_out)}')
                logger.warning(f'Rend_DSR_C - {torch.unique(rend_dsr_c)}')
                logger.warning(f'Rend_DSR_MC - {torch.unique(rend_dsr_mc)}')
                loss_dsr_c[idx] = 0.
                loss_dsr_mc[idx] = 0.

        return loss_dsr_mc.mean(), loss_dsr_c.mean()


    def forward(self, pred, gt, start_dsr):
        pred_cam = pred['pred_cam']
        pred_cam_t = pred['pred_cam_t']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']

        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()

        gt_grph_dsr_mc_dist_mat = gt['grph_dsr_mc_dist_mat']
        gt_grph_dsr_c_label = gt['grph_dsr_c_label']
        gt_grph_dsr_mc_label = gt['grph_dsr_mc_label']
        gt_smpl_textures = gt['smpl_textures_gt']
        gt_valid_labels_dsr_mc = convert_valid_labels(gt['valid_labels_dsr_mc'], 'dsr_mc')
        gt_valid_labels_dsr_c = convert_valid_labels(gt['valid_labels_dsr_c'], 'dsr_c')
        gt_dsr_c_class_weight = gt['dsr_c_class_weight']

        # Compute loss on Semantic rendering
        loss_dsr_mc, loss_dsr_c = self.sr_losses(gt,
            pred_vertices,
            gt_smpl_textures,
            pred_cam_t,
            gt_grph_dsr_mc_dist_mat,
            gt_grph_dsr_c_label,
            gt_grph_dsr_mc_label,
            gt_valid_labels_dsr_mc,
            gt_valid_labels_dsr_c,
            gt_dsr_c_class_weight,
            start_dsr,
        )

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        #logger.info(f'\nl_keypoints - {loss_keypoints}')
        #logger.info(f'l_keypoints_3d - {loss_keypoints_3d}')
        #logger.info(f'dsr_mc_loss - {loss_dsr_mc}')
        #logger.info(f'dsr_c_loss - {loss_dsr_c}')

        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()
        loss_dsr_mc *= self.dsr_mc_loss_weight
        loss_dsr_c *= self.dsr_c_loss_weight


        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
            'loss/loss_dsr_mc': loss_dsr_mc,
            'loss/loss_dsr_c': loss_dsr_c,
        }

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict

class SGCLLoss(nn.Module):
    def __init__(
            self,
            shape_loss_weight=0,
            keypoint_loss_weight=5.,
            pose_loss_weight=1.,
            beta_loss_weight=0.001,
            openpose_train_weight=0.,
            gt_train_weight=1.,
            loss_weight=60.,
            iou_loss_weight=1.0,
            laplacian_loss_weight=0.03,
            sgcl_loss_weight=0.5,
    ):
        super(SGCLLoss, self).__init__()
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()

        self.loss_weight = loss_weight
        self.gt_train_weight = gt_train_weight
        self.pose_loss_weight = pose_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.shape_loss_weight = shape_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.openpose_train_weight = openpose_train_weight
        self.iou_loss_weight = iou_loss_weight
        self.laplacian_loss_weight = laplacian_loss_weight
        self.sgcl_loss_weight = sgcl_loss_weight

    def forward(self, pred, gt):
        pred_cam = pred['pred_cam']
        pred_betas = pred['pred_shape']
        pred_rotmat = pred['pred_pose']
        pred_joints = pred['smpl_joints3d']
        pred_vertices = pred['smpl_vertices']
        pred_projected_keypoints_2d = pred['smpl_joints2d']
        pred_sr_rend_img = pred['sr_rend']
        loss_laplacian = pred['sr_laplacian_loss']

        gt_pose = gt['pose']
        gt_betas = gt['betas']
        gt_joints = gt['pose_3d']
        gt_vertices = gt['vertices']
        gt_keypoints_2d = gt['keypoints']
        has_smpl = gt['has_smpl'].bool()
        has_pose_3d = gt['has_pose_3d'].bool()
        gt_sr_rend_img = gt['grph']

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = smpl_losses(
            pred_rotmat,
            pred_betas,
            gt_pose,
            gt_betas,
            has_smpl,
            criterion=self.criterion_regr,
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = projected_keypoint_loss(
            pred_projected_keypoints_2d,
            gt_keypoints_2d,
            self.openpose_train_weight,
            self.gt_train_weight,
            criterion=self.criterion_keypoints,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(
            pred_joints,
            gt_joints,
            has_pose_3d,
            criterion=self.criterion_keypoints,
        )

        # Per-vertex loss for the shape
        loss_shape = shape_loss(
            pred_vertices,
            gt_vertices,
            has_smpl,
            criterion=self.criterion_shape,
        )

        # compute iou loss with sr_rend and gt_grph
        loss_iou = neg_iou_loss(
            pred_sr_rend_img,
            gt_sr_rend_img,
        )


        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight
        loss_keypoints_3d *= self.keypoint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_betas *= self.beta_loss_weight
        loss_cam = ((torch.exp(-pred_cam[:, 0] * 10)) ** 2).mean()

        loss_iou *= (self.sgcl_loss_weight * self.iou_loss_weight)
        loss_laplacian *= (self.sgcl_loss_weight * self.laplacian_loss_weight)

        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_regr_betas': loss_regr_betas,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
            'loss/loss_iou': loss_iou,
            'loss/loss_laplacian': loss_laplacian,
        }

        loss = sum(loss for loss in loss_dict.values())

        loss *= self.loss_weight

        loss_dict['loss/total_loss'] = loss

        return loss, loss_dict

class GCLLoss(nn.Module):
    def __init__(
        self,
        loss_type='l1',
    ):
        super(GCLLoss, self).__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = self.custom_loss

    def custom_loss(self, pred, gt):
        return torch.sum((pred-gt)**2)

    def forward(self, pred, gt):
        pred_imgs = pred['rend_image']
        gt_imgs = gt['grph']
        loss = self.loss_fn(pred_imgs, gt_imgs)

        loss_dict = {
            'loss/loss_mse': loss,
        }
        return loss, loss_dict

def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        openpose_weight,
        gt_weight,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    loss = (conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss


def keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    loss = criterion(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        has_pose_3d,
        criterion,
):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


def shape_loss(
        pred_vertices,
        gt_vertices,
        has_smpl,
        criterion,
):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


def smpl_losses(
        pred_rotmat,
        pred_betas,
        gt_pose,
        gt_betas,
        has_smpl,
        criterion,
):
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = criterion(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = criterion(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(pred_rotmat.device)
    return loss_regr_pose, loss_regr_betas

def neg_iou_loss(predict, target):

    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims) + 1e-6
    union = (predict + target - predict * target).sum(dims) + 1e-6
    neg_iou = 1. - (intersect / union).sum() / intersect.nelement()

    return neg_iou

