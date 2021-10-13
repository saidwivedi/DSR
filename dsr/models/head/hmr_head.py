import math
import torch
import numpy as np
import torch.nn as nn

from ...core.config import SMPL_MEAN_PARAMS
from ...utils.geometry import rot6d_to_rotmat

class hmr_head(nn.Module):
    def __init__(
            self,
            num_input_features,
            smpl_mean_params=SMPL_MEAN_PARAMS,
    ):
        super(hmr_head, self).__init__()

        npose = 24 * 6
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(num_input_features + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(
            self,
            features,
            init_pose=None,
            init_shape=None,
            init_cam=None,
            n_iter=3
    ):
        batch_size = features.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.avgpool(features)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        }

        return output