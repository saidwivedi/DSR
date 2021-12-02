"""
###########################################################################
# Code adapted from https://github.com/nkolot/SPIN/blob/master/demo.py #
###########################################################################

Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/COCO_val_0544_bbox.json```.

Example with OpenPose detection .json
```
python demo.py --checkpoint=dsr_data/model_checkpoint.pt --img=examples/COCO_val_0544.jpg --openpose=examples/COCO_val_0544_openpose.json
```
Running the previous command will save the results in ```examples/COCO_val_0544_result.png```. It shows both the overlayed and sideview.
"""

import torch
from torchvision.transforms import Normalize
from loguru import logger
import numpy as np
import os
from os.path import isfile, join
import cv2
import argparse
import json

from dsr.models import HMR, SMPL
from dsr.utils.image_utils import crop_cv2
from dsr.utils.renderer import Renderer
from dsr.core import config, constants

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=False, help='Path to input image')
parser.add_argument('--img_folder', type=str, required=False, help='Path to input image folders')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--openpose_folder', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
parser.add_argument('--outfile_folder', type=str, default=None, help='Filename of output images folders')

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop_cv2(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img)[None]
    return norm_img


def output_per_frame(img_fn, bbox_fn, openpose_fn, outfile, model):

    # Preprocess input image and generate predictions
    img = process_image(img_fn, bbox_fn, openpose_fn, input_res=constants.IMG_RES)
    logger.info('Image Processed..')
    
    with torch.no_grad():
        pred = model(img.to(device))
        pred_vertices = pred['smpl_vertices']
        pred_cam_t =pred['pred_cam_t']
        pred_kp_2d = pred['pred_kp2d'] if 'pred_kp2d' in pred.keys() else None

    images_pred = renderer.visualize_tb(
        vertices=pred_vertices,
        camera_translation=pred_cam_t,
        images=img,
        kp_2d=pred_kp_2d,
        sideview=True,
    )
    logger.info('Output mesh rendered')

    outfile = img_fn.split('.')[0] if outfile is None else outfile

    images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
    images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
    outfile = outfile + '_result.png'
    cv2.imwrite(outfile, cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB))
    logger.info(f'Final result saved to {outfile}') 




if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    model = HMR(pretrained=args.checkpoint).to(device)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(
        focal_length=constants.FOCAL_LENGTH,
        img_res=constants.IMG_RES,
        faces=smpl.faces,
    )


    if args.img_folder:
        img_files = [f for f in sorted(os.listdir(args.img_folder)) if isfile(join(args.img_folder, f))]
        openpose_kps = [f.split('.')[0] + '_keypoints.json' for f in img_files]
        outfiles = [f.split('.')[0] + '_result.png' for f in img_files]

        img_files = [join(args.img_folder, f) for f in img_files]
        openpose_kps = [join(args.openpose_folder, f) for f in openpose_kps]
        outfiles = [join(args.outfile_folder, f) for f in outfiles]

        for idx in range(len(img_files)):
            img_fn = img_files[idx]
            openpose_fn = openpose_kps[idx]
            outfile = outfiles[idx]
            output_per_frame(img_fn, None, openpose_fn, outfile, model)
            
    else:
        output_per_frame(args.img, args.bbox, args.openpose, args.outfile, model)

