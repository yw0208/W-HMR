"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import pickle

import numpy as np
import scipy
import torch
from pare.core import constants
from smplx.lbs import vertices2joints
from smplx.utils import Struct, to_tensor, to_np
from smplx.vertex_ids import vertex_ids
from smplx.vertex_joint_selector import VertexJointSelector

from core.cfgs import cfg
from core.constants import H36M_TO_J14
from utils.geometry import projection, perspective_projection


class Graphormer_Body_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''

    def __init__(self, trans_encoder):
        super(Graphormer_Body_Network, self).__init__()
        self.trans_encoder = trans_encoder
        self.global_feat_dim = torch.nn.Linear(2155, 259)
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)

        J_regressor_extra = np.load('/opt/data/private/projects/PyMAF-smpl/data/J_regressor_extra.npy')
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))

        smpl_path = '/opt/data/private/projects/PyMAF-smpl/data/smpl/SMPL_NEUTRAL.pkl'
        with open(smpl_path, 'rb') as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file, encoding='latin1'))
        self.J_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=torch.float32).to('cuda')

        self.vertex_joint_selector = VertexJointSelector(vertex_ids['smplh'])

    def forward(self, ref_feature, grid_feature, smpl_out,orig_shape, temp_verts, J_regressor, meta_masks=None, is_train=False):
        batch_size = ref_feature.size(0)

        # get the last step results 256 x 431 x 3
        ref_vertices = temp_verts
        # ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract global image features and grid features
        global_feat, grid_feat = ref_feature, grid_feature
        # reduce global_feat channel 256 x 2155 -> 256 x 259
        global_feat = self.global_feat_dim(global_feat)
        global_feat = global_feat[:, None, :]
        # process grid features 256 x 256 x 431 -> 256 x 431 x256
        grid_feat = torch.transpose(grid_feat, 1, 2)

        # concatinate template mesh and image feat to form the vertex queries 256 x 431 x 259
        features = torch.cat([ref_vertices, grid_feat], dim=2)
        # prepare input tokens including joint/vertex queries and grid features 256 x 432 x 259
        features = torch.cat([features, global_feat], dim=1)

        if is_train == True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            special_token = torch.ones_like(features[:, :-1, :]).cuda() * 0.01
            features[:, :-1, :] = features[:, :-1, :] * meta_masks + special_token * (1 - meta_masks)

            # forward pass
        # if self.config.output_attentions == True:
        #     features, hidden_states, att = self.trans_encoder(features)
        # else:
        #     features = self.trans_encoder(features)
        features = self.trans_encoder(features)

        pred_vertices_temp = features[:, :-1, :]
        # pred_vertices_temp=pred_vertices_temp+temp_verts

        temp_transpose = pred_vertices_temp.transpose(1, 2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)

        pred_vertices_sub = pred_vertices_sub.transpose(1, 2)
        pred_vertices_full = pred_vertices_full.transpose(1, 2)

        smpl_joints = vertices2joints(self.J_regressor.to(pred_vertices_full.device), pred_vertices_full)
        smpl_joints = self.vertex_joint_selector(pred_vertices_full, smpl_joints)

        extra_joints = vertices2joints(self.J_regressor_extra, pred_vertices_full)
        self_joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        self_joint_map = torch.tensor(self_joints, dtype=torch.long)
        joints = torch.cat([smpl_joints, extra_joints], dim=1)
        joints = joints[:, self_joint_map, :]

        if cfg.TRAIN.STAGE==1:
            joints_2d = projection(joints, smpl_out['pred_cam'])
        else:
            joints_2d = projection(joints.detach(), smpl_out['pred_cam'])

        focal_length = smpl_out['focal_length']
        img_shape = orig_shape[:, [1, 0]]
        camera_center = img_shape / 2.
        if cfg.TRAIN.STAGE==1:
            pred_keypoints_2d_world = perspective_projection(joints.detach(),
                                                             rotation=torch.eye(3, device='cuda').unsqueeze(0).expand(1, -1,
                                                                                                                      -1),
                                                             translation=smpl_out['pred_cam_t'],
                                                             focal_length=focal_length,
                                                             camera_center=camera_center)
        else:
            pred_keypoints_2d_world = perspective_projection(joints,
                                                             rotation=torch.eye(3, device='cuda').unsqueeze(0).expand(1,
                                                                                                                      -1,
                                                                                                                      -1),
                                                             translation=smpl_out['pred_cam_t'],
                                                             focal_length=focal_length,
                                                             camera_center=camera_center)
        pred_keypoints_2d_world_norm = pred_keypoints_2d_world / camera_center.unsqueeze(1) - 1

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices_full)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis
        else:
            pred_joints = joints

        output = {
            'theta': smpl_out['theta'],
            'verts': pred_vertices_full,
            'sub_verts': pred_vertices_sub,
            'temp_verts': pred_vertices_temp,
            'kp_2d': joints_2d,
            'kp_2d_w': pred_keypoints_2d_world_norm,
            'smpl_kp_3d': smpl_joints,
            'kp_3d': pred_joints,
            'rotmat': smpl_out['rotmat'],
            'pred_cam': smpl_out['pred_cam'],
            'pred_cam_t': smpl_out['pred_cam_t'],
            'pred_shape': smpl_out['pred_shape'],
            'pred_pose': smpl_out['pred_pose'],
            'pose': smpl_out['pose'],
            'pelvis': smpl_joints[:, :1, :],
            'focal_length': smpl_out['focal_length'],
            'scale': smpl_out['scale'],
        }

        return output
