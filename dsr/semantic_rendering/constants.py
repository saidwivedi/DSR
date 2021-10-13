import numpy as np

GRPH_LABEL = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 'coat', 'socks',
              'pants', 'jumpsuits', 'scarf', 'skirt', 'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg',
              'leftShoe', 'rightShoe']
GRPH_COLOR_MAP = {'background': [0,0,0], 'hat': [128,0,0], 'hair': [255,0,0], 'glove': [0,85,0], 
                  'sunglasses': [170,0,51], 'upperclothes': [255,85,0], 'dress': [0,0,85], 'coat': [0,119,221], 
                  'socks': [85,85,0], 'pants': [0,85,85], 'jumpsuits': [85,51,0], 'scarf': [52,86,128], 
                  'skirt': [0,128,0], 'face': [0,0,255], 'leftArm': [51,170,221], 'rightArm': [0,255,255], 
                  'leftLeg': [85,255,170], 'rightLeg': [170,255,85], 'leftShoe': [255,255,0], 'rightShoe': [255,170,0]}

GRPH_COLOR_MAP_NORM = {k: np.array(v, dtype=np.float32)/255. for k, v in GRPH_COLOR_MAP.items()}
GRPH_LABEL_IDX = {v: k for k, v in enumerate(GRPH_LABEL)}

GRPH_LABEL_MERGE = ['background', 'head', 'upperclothes', 'pants', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg']
GRPH_MERGE_MAP = {'0': [0], '1': [1, 2, 4, 11, 13], '2': [3, 5, 6, 7, 10], '3': [8, 9, 12], '4': [14],
                  '5': [15], '6': [16, 18], '7': [17, 19]}

CLOTHING_LABEL_MERGE = ['background', 'skin', 'upperclothes', 'pants']
CLOTHING_MERGE_MAP = {'0': [0], '1': [1,2,3,4,8,11,13,14,15,16,17,18,19], '2': [5, 6, 7, 10], '3': [9, 12]}

GRPH_LABEL_TO_SMPL_JOINT = {'leftShoe': [[7,10], [1.0, 1.0]], 
                            'rightShoe': [[8,11], [1.0, 1.0]],
                            'rightArm': [[19,21,23], [1.0, 1.0, 1.0]], 
                            'leftArm': [[18,20,22], [1.0, 1.0, 1.0]],
                            'face': [[12,15], [0.0, 1.0]], 
                            'upperclothes': [[0,3,6,9,13,14,16,17,18,19], [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]],
                            'pants': [[1, 2, 4, 5], [1.0, 1.0, 1.0, 1.0]]}
GRPH_LABEL_TO_KEYPOINTS = {'leftShoe': [14, 19, 20, 21, 30], 'rightShoe': [11, 22, 23, 24, 25]}

#PRETRAINED_CKPT = 'logs/spin/spin_reproduce_latest/02-09-2020_21-22-38_spin_reproduce_latest_training.runsmplify-False/lightning_logs/version_0/checkpoints/epoch=51.ckpt'
PRETRAINED_CKPT = 'data/eft_baseline.pt'
RP_TEXTURE_MAX = '/ps/scratch/ps_shared/sdwivedi/hp_datasets_pseudoGT/renderpeople_train/renderpeople_tex_colors_max_smpl.npy'
RP_TEXTURE_PROB = '/ps/scratch/ps_shared/sdwivedi/hp_datasets_pseudoGT/renderpeople_train/renderpeople_tex_colors_prob_smpl_clean.npy'

SELECTED_GRPH_LABELS = ['leftShoe', 'rightShoe', 'leftArm', 'rightArm', 'face']
#SELECTED_GRPH_LABELS = grph_label
#SELECTED_GRPH_LABELS = ['leftShoe', 'rightShoe', 'leftArm', 'rightArm']
#SELECTED_GRPH_LABELS = ['leftShoe', 'rightShoe']

SRP_LABELS = SELECTED_GRPH_LABELS
#SRP_LABELS = GRPH_LABEL
#SRV_LABELS = CLOTHING_LABEL_MERGE
SRV_LABELS = GRPH_LABEL_MERGE
#SRV_LABELS_MAP = CLOTHING_MERGE_MAP
SRV_LABELS_MAP = GRPH_MERGE_MAP
