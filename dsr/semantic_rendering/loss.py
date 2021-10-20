import torch
import numpy as np
from scipy import ndimage
from loguru import logger

from . import constants

def get_distance_matrix(target):
    dist_mat = ndimage.distance_transform_edt(1-target)
    return dist_mat
    
def distance_transform_loss(predict, dist_mat):
    prod = torch.sum(predict * dist_mat)
    norm = torch.sum(predict) ** (3/2) 
    dist = prod/(norm + 1e-6)
    return dist

def neg_iou_loss(predict, target):
    assert predict.shape == target.shape, 'Target and Predict should have same shape'
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()

def dsr_mc_loss(predict, target, dist_mat, loss_type='DistM', silhouette=False):
    if loss_type == 'DistM':
        return distance_transform_loss(predict[:3], dist_mat)
    elif loss_type == 'nIOU':
        predict = predict[3] if silhouette else predict[:3].mean(0)
        return neg_iou_loss(predict, target[0])
    else:
        logger.warning(f'Not a valid DSR_MC Loss - use DistM/nIOU')
        return 0
