import pickle

import scipy
import torch
import torch.nn as nn
import numpy as np
from pare.utils.geometry import batch_euler2matrix
from pare.utils.train_utils import load_pretrained_model
from smplx.lbs import vertices2joints, batch_rodrigues
from smplx.utils import Struct, to_tensor, to_np
from smplx.vertex_ids import vertex_ids
from smplx.vertex_joint_selector import VertexJointSelector
from timm.models.vision_transformer import Block

from utils.cam_utils import convert_preds_to_angles
from .bert.modeling_graphormer import Graphormer
from .bert.transformers.pytorch_transformers import BertConfig
from .cam_model import CameraRegressorNetwork
from .depth_predictor import Depth_predict_layer
from .e2e_body_network import Graphormer_Body_Network
from .pose_resnet import get_resnet_encoder
from core.cfgs import cfg
from core.path_config import SMPL_Marker
from utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis, unbiased_gram_schmidt, \
    perspective_projection, convert_pare_to_full_img_cam, rotmat_to_rot6d
from .maf_extractor import MAF_Extractor
from .pose_vit import get_vitpose_encoder
from .smpl import SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResNet_Backbone
from .iuv_predictor import IUV_predict_layer
from pare.models.head import HMRHead, SMPLHead, SMPLCamHead
from pare.models import SMPL
from pare.core import config

import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1


class Regressor(nn.Module):
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 9

        self.fc1 = nn.Linear(feat_dim + npose + 13 + 5, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
        self.vertex_joint_selector = VertexJointSelector(vertex_ids['smplh'])

        mean_params = np.load(smpl_mean_params)
        # rot6d to axis angle
        init_pose = torch.from_numpy(mean_params['pose'][:]).reshape(1, 24, 6)
        init_pose = rot6d_to_rotmat(init_pose).reshape(1, -1)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        smpl_path = 'data/smpl/SMPL_NEUTRAL.pkl'
        with open(smpl_path, 'rb') as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file, encoding='latin1'))
        self.J_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=torch.float32)

        smpl_mesh_graph = np.load('data/mesh_downsampling.npz', allow_pickle=True,
                                  encoding='latin1')

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
        Dmap0 = ptD[0].to_dense()
        Dmap1 = ptD[1].to_dense()
        self.register_buffer('Dmap0', Dmap0)
        self.register_buffer('Dmap1', Dmap1)

        self.ssm = np.load(SMPL_Marker)

    def forward(self, x, bbox_info, Tz, orig_shape, center, scale, bbox_height, init_pose=None, init_shape=None,
                init_cam=None, is_train=True, n_iter=1, J_regressor=None):

        x = torch.cat((x, bbox_info), dim=1)
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose.reshape(batch_size, -1)
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = pred_pose.view(batch_size, 24, 3, 3)
        if not is_train:
            pred_rotmat = unbiased_gram_schmidt(pred_rotmat)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        # pred_smpl_joints = pred_output.smpl_joints
        if cfg.TRAIN.STAGE == 1:
            pred_keypoints_2d = projection(pred_joints, pred_cam)
        else:
            pred_keypoints_2d = projection(pred_joints.detach(), pred_cam)

        s = pred_cam[:, 0].detach()
        # print(s.shape,h.shape,Tz.shape)
        focal_length = s * bbox_height * Tz / 2.
        # focal_length = 2200.
        # print(focal_length.shape)
        img_shape = orig_shape[:, [1, 0]]
        camera_center = img_shape / 2.
        pred_cam_t = convert_pare_to_full_img_cam(pred_cam.detach(), bbox_height, center, orig_shape[:, 1],
                                                  orig_shape[:, 0], Tz=Tz)
        if cfg.TRAIN.STAGE == 1:
            pred_keypoints_2d_world = perspective_projection(pred_joints.detach(),
                                                             rotation=torch.eye(3, device='cuda').unsqueeze(0).expand(1,
                                                                                                                      -1,
                                                                                                                      -1),
                                                             translation=pred_cam_t,
                                                             focal_length=focal_length,
                                                             camera_center=camera_center)
        else:
            pred_keypoints_2d_world = perspective_projection(pred_joints,
                                                             rotation=torch.eye(3, device='cuda').unsqueeze(0).expand(1,
                                                                                                                      -1,
                                                                                                                      -1),
                                                             translation=pred_cam_t,
                                                             focal_length=focal_length,
                                                             camera_center=camera_center)

        pred_keypoints_2d_world_norm = pred_keypoints_2d_world / camera_center.unsqueeze(1) - 1
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        sub_verts = torch.matmul(self.Dmap0, pred_vertices)
        temp_verts = torch.matmul(self.Dmap1, sub_verts)
        markers = pred_vertices[:, self.ssm]

        smpl_joints = vertices2joints(self.J_regressor.to(pred_vertices.device), pred_vertices)
        smpl_joints = self.vertex_joint_selector(pred_vertices, smpl_joints)

        output = {
            'theta': torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts': pred_vertices,
            'sub_verts': sub_verts,
            'temp_verts': temp_verts,
            'kp_2d': pred_keypoints_2d,
            'kp_2d_w': pred_keypoints_2d_world_norm,
            'kp_3d': pred_joints,
            'smpl_kp_3d': smpl_joints,
            'rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_cam_t': pred_cam_t,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
            'pose': pose,
            'pelvis': smpl_joints[:, :1, :],
            'scale': scale,
            'focal_length': focal_length,
            'markers': markers
        }
        return output, x

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = pred_pose.view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        # pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        sub_verts = torch.matmul(self.Dmap0, pred_vertices)
        temp_verts = torch.matmul(self.Dmap1, sub_verts)
        markers = pred_vertices[:, self.ssm]

        smpl_joints = vertices2joints(self.J_regressor.to(pred_vertices.device), pred_vertices)
        smpl_joints = self.vertex_joint_selector(pred_vertices, smpl_joints)

        output = {
            'theta': torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts': pred_vertices,
            'sub_verts': sub_verts,
            'temp_verts': temp_verts,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'smpl_kp_3d': smpl_joints,
            'rotmat': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
            'pose': pose,
            'pelvis': smpl_joints[:, :1, :],
            'markers': markers
        }
        return output


class Global_Orient_Regressor(nn.Module):
    def __init__(self, smpl_mean_params):
        super().__init__()
        self.fc1 = nn.Linear(2149 + 6 + 9, 2048)
        # self.fc1 = nn.Linear(2048+6 + 9, 2048)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 2048)
        self.drop2 = nn.Dropout()
        self.decrot = nn.Linear(2048, 9)
        nn.init.xavier_uniform_(self.decrot.weight, gain=0.01)
        mean_params = np.load(smpl_mean_params)
        # rot6d to axis angle
        init_pose = torch.from_numpy(mean_params['pose'][:]).reshape(1, 24, 6)
        init_pose = rot6d_to_rotmat(init_pose).reshape(1, 24, 9)
        init_pose = init_pose[:, 0]
        self.register_buffer('init_pose', init_pose)

    def forward(self, x, cam_rotmat, local_orient, is_train):
        batch_size = x.shape[0]
        cam_rotmat = rotmat_to_rot6d(cam_rotmat)
        init_pose = self.init_pose.expand(batch_size, -1)
        # local_orient=init_pose
        local_orient = local_orient.reshape(batch_size, -1)
        for i in range(3):
            xc = torch.cat([x, cam_rotmat, local_orient], dim=1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_rot = self.decrot(xc) + local_orient
        pred_rot = pred_rot.reshape(-1, 1, 3, 3)
        if not is_train:
            pred_rot = unbiased_gram_schmidt(pred_rot)
        return pred_rot


class WHMR(nn.Module):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True):
        super().__init__()
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            self.global_mode = not cfg.MODEL.PyMAF.MAF_ON
            self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)
        elif cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
            self.feature_extractor = get_vitpose_encoder(cfg)

        # deconv layers
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            self.inplanes = self.feature_extractor.inplanes
        elif cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
            self.inplanes = 768
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.PyMAF.N_ITER):
            self.maf_extractor.append(MAF_Extractor())
        ma_feat_len = 67 * cfg.MODEL.PyMAF.MLP_DIM[-1]

        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            grid_width = 8
            grid_height = 8
        elif cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
            grid_width = 7
            grid_height = 9

        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_width), torch.linspace(-1, 1, grid_height)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_width * grid_height * cfg.MODEL.PyMAF.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(3):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len
            self.regressor.append(Regressor(feat_dim=ref_infeat_dim, smpl_mean_params=smpl_mean_params))

        input_feat_dim = [259]
        hidden_feat_dim = [32]
        output_feat_dim = [3]
        which_blk_graph = [1]
        self.transformer = nn.ModuleList()
        for j in range(3, cfg.MODEL.PyMAF.N_ITER):
            trans_encoder = []
            for i in range(1):
                config_class, model_class = BertConfig, Graphormer
                config = config_class.from_pretrained('models/bert/bert-base-uncased/config.json')

                config.output_attentions = False
                config.hidden_dropout_prob = 0.1
                config.img_feature_dim = input_feat_dim[i]
                config.output_feature_dim = output_feat_dim[i]
                config.hidden_size = hidden_feat_dim[i]
                config.intermediate_size = int(config.hidden_size * 2)

                if which_blk_graph[i] == 1:
                    config.graph_conv = True
                    logger.info("Add Graph Conv")
                else:
                    config.graph_conv = False

                # update model structure if specified in arguments
                config.mesh_type = 'body'
                config.num_hidden_layers = 4
                config.num_attention_heads = 4

                # init a transformer encoder and append it to a list
                assert config.hidden_size % config.num_attention_heads == 0
                model = model_class(config=config)
                logger.info("Init model from scratch.")
                trans_encoder.append(model)

            trans_encoder = torch.nn.Sequential(*trans_encoder)
            self.transformer.append(Graphormer_Body_Network(trans_encoder))

        dp_feat_dim = 256
        self.with_uv = cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)
        if cfg.MODEL.PyMAF.DEPTH_SUPV_ON:
            self.dpth_head = Depth_predict_layer(feat_dim=dp_feat_dim)

        # sub_module to predict Tz
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=7, stride=2, padding=0, bias=False),
                nn.Conv2d(in_channels=64, out_channels=5, kernel_size=7, stride=2, padding=0, bias=False)
            )
            self.transformer_decoder = Block(dim=10 * 10, num_heads=2)
            self.avgpool = nn.AvgPool1d(kernel_size=5)
            self.est_Tz = nn.Sequential(
                nn.Linear(10 * 10, 10),
                nn.Linear(10, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        elif cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=7, stride=3, padding=0, bias=False),
                nn.Conv2d(in_channels=64, out_channels=5, kernel_size=7, stride=2, padding=0, bias=False)
            )
            # self.avgpool = nn.AvgPool2d(kernel_size=16, stride=8),
            self.transformer_decoder = Block(dim=18 * 12, num_heads=2)
            self.avgpool = nn.AvgPool1d(kernel_size=5)
            self.est_Tz = nn.Sequential(
                nn.Linear(18 * 12, 12),
                nn.Linear(12, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )

        self.cam_model = CameraRegressorNetwork(
            backbone='resnet50',
            num_fc_layers=1,
            num_fc_channels=1024,
        )
        ckpt = torch.load('data/pretrained_model/camcalib_sa_biased_l2.ckpt')
        self.cam_model = load_pretrained_model(self.cam_model, ckpt['state_dict'], remove_lightning=True, strict=True)

        self.global_orient = Global_Orient_Regressor(smpl_mean_params)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, meta_masks, center, scale, bbox_height, orig_shape, bbox_info, is_train=False,
                J_regressor=None, full_x=None, cam_rotmat=None):

        batch_size = x.shape[0]

        # get cam_rotmat
        if cam_rotmat is None:
            if full_x is not None:
                pred, _ = self.cam_model(full_x)
                pred = convert_preds_to_angles(
                    *pred, loss_type='softargmax_l2',
                )
                pred_vfov = pred[0]
                pred_pitch = pred[1]
                pred_roll = pred[2]
                cam_pitch = pred_pitch.unsqueeze(-1)
                cam_roll = pred_roll.unsqueeze(-1)
                zeros = torch.zeros((batch_size, 1), device='cuda')
                cam_rotmat = batch_euler2matrix(torch.cat([cam_pitch, zeros, cam_roll], dim=1).float()).detach()
                render_rotmat = batch_euler2matrix(torch.cat([-cam_pitch, zeros, cam_roll], dim=1).float()).detach()
            else:
                cam_rotmat = torch.eye(3, device='cuda').unsqueeze(0).expand(batch_size, -1, -1).float().detach()

        # if full_x is not None:
        #     pred, cam_feat = self.cam_model(full_x)

        # spatial features and global features
        if cfg.MODEL.PyMAF.BACKBONE == 'res50':
            s_feat, g_feat = self.feature_extractor(x)
        elif cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
            s_feat = self.feature_extractor(x)
        # s_feat, g_feat = s_feat.detach(), g_feat.detach()
        # s_feat = x

        assert cfg.MODEL.PyMAF.N_ITER >= 0 and cfg.MODEL.PyMAF.N_ITER <= 3
        if cfg.MODEL.PyMAF.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif cfg.MODEL.PyMAF.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif cfg.MODEL.PyMAF.N_ITER == 3:
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(s_feat, J_regressor=J_regressor)

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []
        out_list['dpth_out'] = []

        # for visulization
        vis_feat_list = [s_feat.detach()]

        # deconv s_feat
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            s_feat_i = deconv_blocks[rf_i](s_feat)
            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach())
            self.maf_extractor[rf_i].im_feat = s_feat_i

        # Tz predictions
        if cfg.TRAIN.STAGE == 1:
            s_feat_mini = self.conv(s_feat.detach())
        else:
            s_feat_mini = self.conv(s_feat)
        s_feat_mini = s_feat_mini.reshape(batch_size, 5, -1)
        # bbox_info_Tz=bbox_info[:,:,None]
        # s_feat_mini=torch.cat((s_feat_mini,bbox_info_Tz),dim=2)
        s_feat_decoder = self.transformer_decoder(s_feat_mini).transpose(1, 2)
        s_feat_short = self.avgpool(s_feat_decoder).squeeze(-1)
        # s_feat_short = torch.cat((s_feat_short, bbox_info), dim=1)
        Tz = 10. * self.est_Tz(s_feat_short).squeeze(-1)

        # parameter predictions
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            pred_cam = smpl_output['pred_cam']
            pred_shape = smpl_output['pred_shape']
            pred_pose = smpl_output['rotmat']
            temp_verts = smpl_output['temp_verts']
            markers = smpl_output['markers']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()
            temp_verts = temp_verts.detach()
            markers = markers.detach()

            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2)
                ref_feature, grid_feature = self.maf_extractor[rf_i].sampling(sample_points)
                smpl_output, _ = self.regressor[rf_i](ref_feature, bbox_info, Tz, orig_shape, center, scale,
                                                      bbox_height,
                                                      pred_pose,
                                                      pred_shape, pred_cam,
                                                      is_train, n_iter=1, J_regressor=J_regressor, )
            else:
                focal_length = smpl_output['focal_length']
                img_center = orig_shape[:, [1, 0]] / 2.
                ref_feature, grid_feature = self.maf_extractor[rf_i](markers, center, scale,
                                                                     focal_length.detach(), img_center)
                smpl_output, body_feat = self.regressor[rf_i](ref_feature, bbox_info, Tz, orig_shape, center, scale,
                                                              bbox_height,
                                                              pred_pose, pred_shape,
                                                              pred_cam, is_train, n_iter=1,
                                                              J_regressor=J_regressor)
                # if rf_i == 1:
                #     ref_feature, grid_feature = self.maf_extractor[rf_i](markers, center, scale,
                #                                                          focal_length.detach(), img_center)
                #     smpl_output = self.regressor[rf_i](ref_feature, bbox_info, Tz, orig_shape, center, scale,
                #                                        bbox_height,
                #                                        pred_pose, pred_shape,
                #                                        pred_cam, is_train, n_iter=1,
                #                                        J_regressor=J_regressor)
                # else:
                #     ref_feature, grid_feature = self.maf_extractor[rf_i](temp_verts, center, scale,
                #                                                          focal_length.detach(), img_center)
                #     smpl_output = self.transformer[0](ref_feature, grid_feature, smpl_output, orig_shape, temp_verts,
                #                                       J_regressor,
                #                                       meta_masks, is_train)
            out_list['smpl_out'].append(smpl_output)
            # print(smpl_output['kp_2d_w'])

        last_local_rotmat = smpl_output['rotmat'][:, 0]
        global_rotmat = self.global_orient(body_feat, cam_rotmat, last_local_rotmat, is_train)
        global_axix_angle = rotation_matrix_to_angle_axis(global_rotmat.reshape(-1, 3, 3)).reshape(-1, 3)
        global_pose = torch.cat([global_axix_angle, smpl_output['pose'][:, 3:]], dim=1)
        global_shape = smpl_output['pred_shape']
        global_output = {}
        global_output['global_pose'] = global_pose
        global_output['global_shape'] = global_shape

        global_output['global_rotmat'] = torch.cat([global_rotmat, smpl_output['rotmat'][:, 1:]], dim=1)
        # global_output['global_rotmat'] = smpl_output['rotmat']
        global_smpl_output = self.regressor[0].smpl(betas=smpl_output['pred_shape'],
                                                    body_pose=global_output['global_rotmat'][:, 1:],
                                                    global_orient=global_output['global_rotmat'][:, 0].unsqueeze(1),
                                                    pose2rot=False, )
        pred_vertices = global_smpl_output.vertices
        pred_joints = global_smpl_output.joints
        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis
        global_output['global_kp_3d'] = pred_joints
        global_output['global_verts'] = pred_vertices
        out_list['global_output'] = global_output

        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            iuv_out_dict = self.dp_head(s_feat)
            out_list['dp_out'].append(iuv_out_dict)

        if cfg.MODEL.PyMAF.DEPTH_SUPV_ON:
            depth_image = self.dpth_head(s_feat)
            out_list['dpth_out'].append(depth_image)
        output_dict = {}
        output_dict['global_output'] = out_list['global_output']
        vis_dict = {
            'local_smpl_vertices': smpl_output['verts'],
            'smpl_vertices': pred_vertices,
            'pred_cam_t': smpl_output['pred_cam_t'],
            'focal_length': smpl_output['focal_length'],
            'cam_rotmat': cam_rotmat,
            'render_rotmat': render_rotmat,
            'shape': global_shape,
            'global_pose': global_pose,
            'local_pose': smpl_output['pose']
        }
        # return output_dict
        # return out_list, vis_feat_list
        return vis_dict


def whmr_net(smpl_mean_params, pretrained=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WHMR(smpl_mean_params, pretrained)
    return model
