import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
#    if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import cv2
import torch
import trimesh
import pyrender
import numpy as np
from torchvision.utils import make_grid
from ..core import constants

from . import kp_utils

def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),
        'light_pink': np.array([233, 163, 201]),
        'light_green': np.array([161, 215, 106]),
        'green': np.array([77, 146, 33]),
        'red': np.array([215, 48, 39]),
        'light_red': np.array([252, 146, 114]),
        'light_orange': np.array([252, 141, 89]),
        'purple': np.array([118, 42, 131]),
        'light_purple': np.array([175, 141, 195]),
        'light_blue': np.array([145, 191, 219]),
        'blue': np.array([69, 117, 180]),
        'gray': np.array([130, 130, 130]),
        'white': np.array([255, 255, 255]),
        'pinkish': np.array([204, 77, 77]),
    }
    return colors

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None, mesh_color='blue'):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.mesh_color = get_colors()[mesh_color]

    def de_norm(self, images):
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        return images

    def visualize_tb(self, vertices, camera_translation, images, kp_2d=None,\
                     nb_max_img=6, sideview=False):

        images = self.de_norm(images)
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))

        if kp_2d is not None:
            kp_2d = kp_2d.cpu().numpy()

        rend_imgs = []
        nb_max_img = min(nb_max_img, vertices.shape[0])
        for i in range(nb_max_img):

            rend_img = torch.from_numpy(
                np.transpose(self.__call__(vertices[i],
                             camera_translation[i], 
                             images_np[i]),
                (2,0,1))
            ).float()

            rend_imgs.append(images[i])
            if kp_2d is not None:
                kp_img = draw_skeleton(images_np[i].copy(), kp_2d=kp_2d[i], dataset='smpl')
                kp_img = torch.from_numpy(np.transpose(kp_img, (2,0,1))).float()
                rend_imgs.append(kp_img)

            rend_imgs.append(rend_img)

            if sideview:
                side_img = torch.from_numpy(
                    np.transpose(
                        self.__call__(vertices[i], camera_translation[i], np.ones_like(images_np[i]), sideview=True),
                        (2,0,1)
                    )
                ).float()
                rend_imgs.append(side_img)

        nrow = 1
        if sideview: nrow += 1
        nrow += 1
        if kp_2d is not None: nrow += 1

        rend_imgs = make_grid(rend_imgs, nrow=nrow)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, sideview=False):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(self.mesh_color[0] / 255., self.mesh_color[1] / 255., self.mesh_color[2] / 255., 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        if sideview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(270), [0, 1, 0])
            mesh.apply_transform(rot)

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img


def draw_skeleton(image, kp_2d, dataset='common', unnormalize=True, thickness=2):
    image = image * 255
    image = np.clip(image, 0, 255)

    if unnormalize:
        kp_2d[:,:2] = 0.5 * 224 * (kp_2d[:, :2] + 1) # normalize_2d_kp(kp_2d[:,:2], 224, inv=True)

    kp_2d = np.hstack([kp_2d, np.ones((kp_2d.shape[0], 1))])

    kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)


    rcolor = [255,0,0]
    pcolor = [0,255,0]
    lcolor = [0,0,255]

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()

    # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx, pt in enumerate(kp_2d):
        # if pt[2] > 0: # if visible
        cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
        # cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

    for i,(j1,j2) in enumerate(skeleton):
        # if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
        # if dataset == 'common':
        #     color = rcolor if common_lr[i] == 0 else lcolor
        # else:
        color = lcolor if i % 2 == 0 else rcolor
        pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
        cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    image = np.asarray(image, dtype=float) / 255
    return image


def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d
