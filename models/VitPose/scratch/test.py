import torch
from mmcv import Config
from torchsummary import summary

from mmpose.models import build_backbone

if __name__ == '__main__':
    config = '/home/yaowei/project/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
    cfg = Config.fromfile(config)
    backbone = build_backbone(cfg.backbone).cuda()
    model = build_backbone(cfg.model)
    input = torch.randn((1, 3, 256, 192)).cuda()
    output = backbone(input)
    print(output.shape)
