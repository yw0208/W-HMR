from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class Depth_predict_layer(nn.Module):
    def __init__(self, feat_dim=256, final_cov_k=3):
        super().__init__()

        self.predict_depth = nn.Conv2d(
            in_channels=feat_dim,
            out_channels=1,
            kernel_size=final_cov_k,
            stride=1,
            padding=1 if final_cov_k == 3 else 0
        )

    def forward(self, x):

        predict_depth = self.predict_depth(x)

        return predict_depth