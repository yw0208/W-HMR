import torch
from mmcv import Config
from torch import nn

from models.ViTPose.mmpose.models import build_backbone


class VitPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config.backbone)

    def forward(self, x):
        return self.backbone(x)


def get_vitpose_encoder(cfg=None):
    config = 'models/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
    config = Config.fromfile(config)
    model = VitPose(config)
    checkpoint = torch.load('data/pretrained_model/vitpose-b-multi-coco.pth')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model
