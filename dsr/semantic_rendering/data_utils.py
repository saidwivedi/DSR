# Preparing GT Graphonomy Texture

import torch
import numpy as np

from loguru import logger
from . import constants

def inverse_ratio(arr):
    inv_ratio = np.ones(arr.shape[0], dtype=np.float32) * 1e-5
    total_sum = arr.sum()
    for i in range(inv_ratio.shape[0]):
        if arr[i] > 60:
            inv_ratio[i] = 1 - (arr[i]/total_sum)
    return inv_ratio

def get_class_ratio(arr, threshold=60):
    cls_ratio = np.zeros(arr.shape[0], dtype=np.float32)
    thres_index = ((arr - threshold) >= 0).nonzero()[0]
    if len(thres_index) ==  0:
        return cls_ratio
    else:
        smallest_num = np.sort(arr[thres_index])[0]
    for i in range(cls_ratio.shape[0]):
        if arr[i] > threshold:
            cls_ratio[i] = smallest_num / arr[i]
    return cls_ratio

def remove_background_from_keypoints(img, keypoints):
    # Remove background persons using keypoints
    margin = 30
    keypoints_valid = keypoints[0][np.where(keypoints[0,:,2] > 0.3)]
    xmin, xmax = int(max(keypoints_valid[:,0].min()-margin, 0)), \
                 int(min(keypoints_valid[:,0].max()+margin, img.shape[0]))
    ymin, ymax = int(max(keypoints_valid[:,1].min()-margin, 0)), \
                 int(min(keypoints_valid[:,1].max()+margin, img.shape[1]))
    img[0:ymin], img[ymax:] = 0.0, 0.0
    img[:,0:xmin], img[:,xmax:] = 0.0, 0.0
    return img

def convert_fixed_length_vector(valid_labels, method='dsr_mc'):
    if method == 'dsr_mc':
        labels = constants.DSR_MC_LABELS
    elif method == 'dsr_c':
        labels = constants.DSR_C_LABELS
    label_vector = np.zeros((len(labels)), dtype=np.int)
    valid_index = [labels.index(val_label) for val_label in valid_labels]
    label_vector[valid_index] = 1
    return label_vector

def convert_valid_labels(label_vector, method='dsr_mc'):
    if method == 'dsr_mc':
        labels = constants.DSR_MC_LABELS
    elif method == 'dsr_c':
        labels = constants.DSR_C_LABELS

    valid_labels = []
    for sample in range(label_vector.shape[0]):
        valid_labels.append([labels[i] for i,j in enumerate(label_vector[sample]) if j == 1])
    return valid_labels

# Get probabilistic labels as texture for DSR_MC
def get_dsr_mc_probPrior(valid_labels=None):
    rp_textures_prob = np.squeeze(np.load(constants.RP_TEXTURE_PROB)) # -- 13776 x 20
    smpl_textures_gcl_gt = np.zeros(rp_textures_prob.shape[0], dtype=np.float32)

    for sel_part in valid_labels:
        valid_label_id = constants.GRPH_LABEL_IDX[sel_part]
        smpl_textures_gcl_gt += rp_textures_prob[:, valid_label_id]
    smpl_textures_gcl_gt = np.repeat(smpl_textures_gcl_gt[:, np.newaxis], 3, axis=1)

    return smpl_textures_gcl_gt

def get_dsr_c_probPrior(merge=True, merge_map=None):
    rp_textures_gcl = np.squeeze(np.load(constants.RP_TEXTURE_PROB)) # -- 13776 x 20

    if merge == True:
        rp_textures_gcl_merge = np.zeros((rp_textures_gcl.shape[0], len(merge_map.keys())), dtype=np.float32)
        for key, values in merge_map.items():
            for value in values:
                rp_textures_gcl_merge[:, int(key)] += rp_textures_gcl[:, value]
        rp_textures_gcl = rp_textures_gcl_merge

    rp_textures_gcl = np.repeat(rp_textures_gcl[:, :, np.newaxis], 3, axis=2)
    rp_textures_gcl = np.swapaxes(rp_textures_gcl, 0, 1)
    return rp_textures_gcl

def convert_grph_to_labels(grph, keypoints=None, merge=True, merge_map=None, merge_label=None):

    grph = remove_background_from_keypoints(grph, keypoints)

    new_grph = np.zeros_like(grph, dtype=np.float32)
    valid_labels = constants.GRPH_LABEL
    num_pixels_per_label = np.zeros(len(constants.GRPH_LABEL), dtype=np.float32)

    for idx, sel_part in enumerate(valid_labels):
        valid_index = np.sum(grph == constants.GRPH_COLOR_MAP[sel_part], axis=2) == 3
        num_pixels = np.flatnonzero(valid_index).size

        if num_pixels > 0: # Thresholding number of pixels for each label (keep threshold 0 for now)
            #print(f'Number of pixels in {sel_part} -> {num_pixels}')
            num_pixels_per_label[idx] = num_pixels
            new_grph[valid_index] = np.array([idx, idx, idx])

    if merge == True:
        merge_grph = np.zeros_like(grph, dtype=np.float32)
        merge_num_pixels_per_label = np.zeros(len(merge_map.keys()), dtype=np.float32)
        for key, values in merge_map.items():
            for value in values:
                merge_grph[new_grph == value] = int(key)
                merge_num_pixels_per_label[int(key)] += num_pixels_per_label[value]

        new_grph, num_pixels_per_label = merge_grph, merge_num_pixels_per_label
        valid_labels = merge_label

    new_grph = new_grph[:,:,0]

    cls_ratio = get_class_ratio(num_pixels_per_label)

    return new_grph, valid_labels, cls_ratio

def convert_grph_to_binary_mask(grph, norm=False, only_valid_labels=True, keypoints=None):

    grph = remove_background_from_keypoints(grph, keypoints)

    # Convert graphonomy labels to binary mask (with threshold pixels)
    valid_labels, num_pixels_per_label = [], []
    binary_mask = 255. * np.array([1., 1., 1.], dtype=np.float32)
    new_grph = np.zeros_like(grph)

    for sel_part in constants.DSR_MC_LABELS:
        if sel_part == 'background':
            continue
        valid_index = np.sum(grph == constants.GRPH_COLOR_MAP[sel_part], axis=2) == 3
        num_pixels = np.flatnonzero(valid_index).size
        #print(f'Number of pixels in {sel_part} -> {num_pixels}')

        if num_pixels > 60: # Thresholding number of pixels for each label
            new_grph[valid_index] = binary_mask
            valid_labels.append(sel_part)
            num_pixels_per_label.append(num_pixels)

    if norm == True:
        new_grph = new_grph/255.
    if only_valid_labels == False:
        valid_labels = constants.DSR_MC_LABELS

    return new_grph, valid_labels, np.array(num_pixels_per_label)


def convert_camT_to_proj_mat(cam_t, FOCAL_L=5000., IMG_SIZE=224):
    R = torch.eye(3)
    RT = torch.zeros((cam_t.shape[0],3,4)) # camera extrinsic parameter
    RT[:,:,0:3] = R
    RT[:,:,3] = cam_t
    RT[:,1,1] *= -1.0 # Rotate around Y axis for NMR rendering
    RT[:,1,3] *= -1.0 # Negate the y translation for NMR rendering
    K = torch.Tensor([[FOCAL_L, 0, IMG_SIZE/2],
                      [0, FOCAL_L, IMG_SIZE/2],
                      [0, 0, 1]]) # camera intrinsic parameter
    P = torch.matmul(K, RT)
    return P

def get_conf_mask(valid_labels):
    conf_mask = np.zeros((1,49))
    for label in valid_labels:
        conf_mask[0][GRPH_LABELS_TO_KEYPOINTS[label]] = 1.0
    return conf_mask

def fix_seed(seed):
    print(f'Seed value for the experiment -> {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
