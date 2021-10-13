import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import hmr_head, smpl_head
from .backbone.utils import get_backbone_info
from ..utils.train_utils import add_smpl_params_to_dict, prepare_statedict


class HMR(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            img_res=224,
            pretrained=None,
    ):
        super(HMR, self).__init__()
        self.backbone = eval(backbone)(pretrained=True)
        self.head = hmr_head(
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
        )
        self.smpl = smpl_head(img_res=img_res)
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(self, images):
        features = self.backbone(images)
        hmr_output = self.head(features)
        smpl_output = self.smpl(
            rotmat=hmr_output['pred_pose'],
            shape=hmr_output['pred_shape'],
            cam=hmr_output['pred_cam'],
            normalize_joints2d=True,
        )
        smpl_output.update(hmr_output)
        return smpl_output

    def load_pretrained(self, file):
        logger.info(f'Loading pretrained weights from {file}')
        try:
            state_dict = torch.load(file)['model']
        except:
            try:
                state_dict = prepare_statedict(torch.load(file)['state_dict'])
            except:
                state_dict = add_smpl_params_to_dict(torch.load(file))
        self.backbone.load_state_dict(state_dict, strict=False)
        self.head.load_state_dict(state_dict, strict=False)
