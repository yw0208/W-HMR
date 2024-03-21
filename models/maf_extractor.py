# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py
from packaging import version
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.cfgs import cfg
from utils.geometry import projection

import logging

logger = logging.getLogger(__name__)


class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator

    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, device=torch.device('cuda')):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        filter_channels = cfg.MODEL.PyMAF.MLP_DIM
        self.last_op = nn.ReLU(True)

        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            self.add_module("conv%d" % l, self.filters[l])

        self.im_feat = None
        self.cam = None

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load('data/mesh_downsampling.npz', allow_pickle=True, encoding='latin1')

        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D']  # shape: (2,)

        # downsampling
        ptD = []
        for i in range(len(D)):
            d = scipy.sparse.coo_matrix(D[i])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense())  # 6890 -> 431
        self.register_buffer('Dmap', Dmap)

        self.crop_size = cfg.IMG_RES.WIDTH

    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        y = self.last_op(y)

        y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None, z_feat=None):
        '''
        Given 2D points, sample the point-wise features for each point,
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        if im_feat is None:
            im_feat = self.im_feat

        batch_size = im_feat.shape[0]

        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            # Default grid_sample behavior has changed to align_corners=False since 1.3.0.
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]
        else:
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0]

        mesh_align_feat = self.reduce_dim(point_feat)
        return mesh_align_feat, point_feat

    def forward(self, p, center, scale, img_focal, img_center, s_feat=None, cam=None, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        if cam is None:
            cam = self.cam
        p_proj_2d = projection(p, cam, retain_z=False)

        # p_proj_2d = self.project(p, cam, center, scale, img_focal, img_center)
        # p_proj_2d = p_proj_2d.detach().to(torch.float32)
        mesh_align_feat, point_feat = self.sampling(p_proj_2d, s_feat)
        return mesh_align_feat, point_feat

    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):

        trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

        # Projection in full frame image coordinate
        points = points + trans_full
        points2d_full = self.perspective_projection(points, rotation=None, translation=None,
                                                    focal_length=img_focal, camera_center=img_center)

        # Adjust projected points to crop image coordinate
        # (s.t. 1. we can calculate loss in crop image easily
        #       2. we can query its pixel in the crop
        #  )
        b = scale * 200
        points2d = points2d_full - (center - b[:, None] / 2)[:, None, :]
        points2d = points2d * (self.crop_size / b)[:, None, None]

        bbox_size = torch.tensor([cfg.IMG_RES.WIDTH, cfg.IMG_RES.HEIGHT], device=points2d.device) / 2.
        points2d = (points2d - bbox_size) / bbox_size

        if return_full:
            return points2d_full, points2d
        else:
            return points2d

    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        b = scale * 200
        cx, cy = center[:, 0], center[:, 1]  # center of crop
        s, tx, ty = pred_cam.unbind(-1)

        img_cx, img_cy = img_center[:, 0], img_center[:, 1]  # center of original image

        bs = b * s
        tx_full = tx + 2 * (cx - img_cx) / bs
        ty_full = ty + 2 * (cy - img_cy) / bs
        tz_full = 2 * img_focal / bs

        trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
        trans_full = trans_full.unsqueeze(1)

        return trans_full

    def perspective_projection(self, points, rotation, translation,
                               focal_length, camera_center, distortion=None):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points.shape[0]

        # Extrinsic
        if rotation is not None:
            points = torch.einsum('bij,bkj->bki', rotation, points)

        if translation is not None:
            points = points + translation.unsqueeze(1)

        if distortion is not None:
            kc = distortion
            points = points[:, :, :2] / points[:, :, 2:]

            r2 = points[:, :, 0] ** 2 + points[:, :, 1] ** 2
            dx = (2 * kc[:, [2]] * points[:, :, 0] * points[:, :, 1]
                  + kc[:, [3]] * (r2 + 2 * points[:, :, 0] ** 2))

            dy = (2 * kc[:, [3]] * points[:, :, 0] * points[:, :, 1]
                  + kc[:, [2]] * (r2 + 2 * points[:, :, 1] ** 2))

            x = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 0] + dx
            y = (1 + kc[:, [0]] * r2 + kc[:, [1]] * r2.pow(2) + kc[:, [4]] * r2.pow(3)) * points[:, :, 1] + dy

            points = torch.stack([x, y, torch.ones_like(x)], dim=-1)

        # Intrinsic
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Apply camera intrinsicsrf
        points = points / points[:, :, -1].unsqueeze(-1)
        projected_points = torch.einsum('bij,bkj->bki', K, points.to(torch.float32))
        projected_points = projected_points[:, :, :-1]

        return projected_points
