# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/train/trainer.py

import time

import joblib
import scipy
import smplx
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from skimage.transform import resize
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from pare.models import SMPL
from datasets.data_utils.ThreeDPW_dataset import TCMRDataset
from .base_trainer import BaseTrainer
from datasets import MixedDataset, BaseDataset
from models import hmr, whmr_net
from utils.pose_utils import compute_similarity_transform_batch
from utils.geometry import perspective_projection, estimate_translation, batch_rodrigues
from smplx.lbs import vertices2joints
from core import path_config, constants
from .fits_dict import FitsDict
from .cfgs import cfg
from utils.train_utils import print_args
from utils.iuvmap import iuv_img2map, iuv_map2img

# try:
#     from utils.renderer import PyRenderer, IUV_Renderer
# except:
#     print('fail to import Renderer.')
from utils.renderer import IUV_Renderer, Depth_Renderer
import logging

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def init_fn(self):
        if self.options.rank == 0:
            self.summary_writer.add_text('command_args', print_args())

        if self.options.regressor == 'hmr':
            # HMR/SPIN modelpy
            self.model = hmr(path_config.SMPL_MEAN_PARAMS, pretrained=True)
            self.smpl = SMPL(path_config.SMPL_MODEL_DIR,
                             batch_size=cfg.TRAIN.BATCH_SIZE,
                             create_transl=False).to(self.device)
        elif self.options.regressor == 'pymaf_net':
            # PyMAF model
            self.smpl_male = SMPL(model_path=path_config.SMPL_MODEL_DIR,
                                  gender='male',
                                  create_transl=False).to(self.device)
            self.smpl_female = SMPL(model_path=path_config.SMPL_MODEL_DIR,
                                    gender='female',
                                    create_transl=False).to(self.device)
            self.model = whmr_net(path_config.SMPL_MEAN_PARAMS, pretrained=True)
            self.smpl = self.model.regressor[0].smpl
            self.smplx = smplx.create('/opt/data/private/projects/PyMAF-smpl/data', model_type='smpl', gender='neutral',
                                      ext='npz').to(self.device)

        if self.options.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.options.gpu is not None:
                torch.cuda.set_device(self.options.gpu)
                self.model.cuda(self.options.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.options.batch_size = int(self.options.batch_size / self.options.ngpus_per_node)
                self.options.workers = int(
                    (self.options.workers + self.options.ngpus_per_node - 1) / self.options.ngpus_per_node)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.options.gpu],
                                                                       output_device=self.options.gpu,
                                                                       find_unused_parameters=True)
            else:
                self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
            self.models_dict = {'model': self.model.module}
        else:
            self.model = self.model.to(self.device)
            self.models_dict = {'model': self.model}

        cudnn.benchmark = True

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_focal = nn.MSELoss().to(self.device)
        self.focal_length = constants.FOCAL_LENGTH

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=cfg.SOLVER.BASE_LR,
                                          weight_decay=0)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                                       patience=5, verbose=True)

        self.optimizers_dict = {'optimizer': self.optimizer}

        if self.options.single_dataset:
            self.train_ds = BaseDataset(self.options, self.options.single_dataname, is_train=True)
        else:
            self.train_ds = MixedDataset(self.options, is_train=True)

        self.valid_ds = BaseDataset(self.options, self.options.eval_dataset, is_train=False)

        if self.options.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds)
            val_sampler = None
        else:
            train_sampler = None
            val_sampler = None

        self.train_data_loader = DataLoader(
            self.train_ds,
            batch_size=self.options.batch_size,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=cfg.TRAIN.PIN_MEMORY,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            prefetch_factor=4
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_ds,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=cfg.TRAIN.PIN_MEMORY,
            sampler=val_sampler,
            prefetch_factor=4
        )

        # Load dictionary of fits
        # self.fits_dict = FitsDict(self.options, self.train_ds)
        self.evaluation_accumulators = dict.fromkeys(
            ['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts', 'target_verts'])

        # Create renderer
        try:
            self.renderer = PyRenderer()
        except:
            print('No renderer for visualization.')
            self.renderer = None

        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            self.iuv_maker = IUV_Renderer(orig_size=(cfg.IMG_RES.HEIGHT, cfg.IMG_RES.WIDTH),
                                          output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE)
            self.depth_maker = Depth_Renderer(orig_size=(cfg.IMG_RES.HEIGHT, cfg.IMG_RES.WIDTH),
                                              output_size=cfg.MODEL.PyMAF.DP_HEATMAP_SIZE)

        self.decay_steps_ind = 1
        self.decay_epochs_ind = 1

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
        Dmap0 = ptD[0].to_dense()
        Dmap1 = ptD[1].to_dense()
        self.Dmap0 = Dmap0.to('cuda')
        self.Dmap1 = Dmap1.to('cuda')

        self.save_step=1300

    # def finalize(self):
    #     self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1]))
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d]
        conf = conf[has_pose_3d]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl]
        gt_vertices_with_shape = gt_vertices[has_smpl]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl]
        pred_betas_valid = pred_betas[has_smpl]
        gt_betas_valid = gt_betas[has_smpl]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def body_uv_losses(self, u_pred, v_pred, index_pred, ann_pred, uvia_list, has_iuv=None):
        batch_size = index_pred.size(0)
        device = index_pred.device

        Umap, Vmap, Imap, Annmap = uvia_list

        if has_iuv is not None:
            if torch.sum(has_iuv.float()) > 0:
                u_pred = u_pred[has_iuv] if u_pred is not None else u_pred
                v_pred = v_pred[has_iuv] if v_pred is not None else v_pred
                index_pred = index_pred[has_iuv] if index_pred is not None else index_pred
                ann_pred = ann_pred[has_iuv] if ann_pred is not None else ann_pred
                Umap, Vmap, Imap = Umap[has_iuv], Vmap[has_iuv], Imap[has_iuv]
                Annmap = Annmap[has_iuv] if Annmap is not None else Annmap
            else:
                return (torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device),
                        torch.zeros(1).to(device))

        Itarget = torch.argmax(Imap, dim=1)
        Itarget = Itarget.view(-1).to(torch.int64)

        index_pred = index_pred.permute([0, 2, 3, 1]).contiguous()
        index_pred = index_pred.view(-1, Imap.size(1))

        loss_IndexUV = F.cross_entropy(index_pred, Itarget)

        if cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0:
            loss_U = F.smooth_l1_loss(u_pred[Imap > 0], Umap[Imap > 0], reduction='sum') / batch_size
            loss_V = F.smooth_l1_loss(v_pred[Imap > 0], Vmap[Imap > 0], reduction='sum') / batch_size

            loss_U *= cfg.LOSS.POINT_REGRESSION_WEIGHTS
            loss_V *= cfg.LOSS.POINT_REGRESSION_WEIGHTS
        else:
            loss_U, loss_V = torch.zeros(1).to(device), torch.zeros(1).to(device)

        if ann_pred is None:
            loss_segAnn = None
        else:
            Anntarget = torch.argmax(Annmap, dim=1)
            Anntarget = Anntarget.view(-1).to(torch.int64)
            ann_pred = ann_pred.permute([0, 2, 3, 1]).contiguous()
            ann_pred = ann_pred.view(-1, Annmap.size(1))
            loss_segAnn = F.cross_entropy(ann_pred, Anntarget)

        return loss_U, loss_V, loss_IndexUV, loss_segAnn

    def depth_losses(self, dpth_pred, depth_image_gt, valid_fit):
        batch_size = dpth_pred.shape[0]
        device = dpth_pred.device

        if valid_fit is not None:
            if torch.sum(valid_fit.float()) > 0:
                dpth_pred = dpth_pred[valid_fit] if dpth_pred is not None else dpth_pred
                depth_image_gt = depth_image_gt[valid_fit] if depth_image_gt is not None else depth_image_gt
            else:
                return (torch.zeros(1).to(device))
        if cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0:
            loss_Depth = F.smooth_l1_loss(dpth_pred, depth_image_gt, reduction='sum') / batch_size
            loss_Depth *= cfg.LOSS.POINT_REGRESSION_WEIGHTS
        else:
            loss_Depth = torch.zeros(1).to(device)

        return loss_Depth

    def focal_losses(self, focal_pred, focal_gt):
        return self.criterion_focal(focal_pred, focal_gt)

    def train(self, epoch):
        """Training process."""
        if self.options.distributed:
            self.train_data_loader.sampler.set_epoch(epoch)

        self.model.train()

        # Learning rate decay
        if self.decay_epochs_ind < len(cfg.SOLVER.EPOCHS) and epoch == cfg.SOLVER.EPOCHS[self.decay_epochs_ind]:
            lr = self.optimizer.param_groups[0]['lr']
            lr_new = lr * cfg.SOLVER.GAMMA
            print('Decay the learning on epoch {} from {} to {}'.format(epoch, lr, lr_new))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_new
            lr = self.optimizer.param_groups[0]['lr']
            assert lr == lr_new
            self.decay_epochs_ind += 1

        if self.options.rank == 0:
            pbar = tqdm(desc=self.options.log_name + ' Epoch ' + str(epoch),
                        total=len(self.train_ds) // cfg.TRAIN.BATCH_SIZE,
                        initial=self.checkpoint_batch_idx)

        # Iterate over all batches in an epoch
        for step, batch in enumerate(self.train_data_loader, self.checkpoint_batch_idx):
            if self.options.rank == 0:
                pbar.update(1)

            self.step_count += 1
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = self.train_step(batch)

            if self.step_count % 5000 == 0 and self.options.rank == 0:
                if self.options.single_dataset and self.options.single_dataname == 'surreal':
                    self.validate()
                    performance = self.evaluate()
                    is_best = performance < self.best_performance
                    if is_best:
                        logger.info('Best performance achived, saved it!')
                        self.best_performance = performance
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                               cfg.TRAIN.BATCH_SIZE,
                                               self.step_count, is_best, save_by_step=True)

            # Tensorboard logging every summary_steps steps
            if self.step_count % cfg.TRAIN_VIS_ITER_FERQ == 0 and self.options.rank == 0:
                self.model.eval()
                # self.visualize(self.step_count, batch, 'train', **out)
                self.model.train()

        if self.options.rank == 0:
            pbar.close()

        # load a checkpoint only on startup, for the next epochs
        # just iterate over the dataset as usual
        self.checkpoint = None
        return

    def train_step(self, input_batch):
        self.model.train()
        # Get data from the batch
        images = input_batch['img']  # input image
        gt_keypoints_2d = input_batch['keypoints']  # 2D keypoints
        gt_keypoints_2d_world = input_batch['keypoints_world']  # 2D keypoints in world coordinate
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d']  # 3D pose
        has_smpl = input_batch['has_smpl'].to(torch.bool)  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].to(torch.bool)  # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        indices = input_batch['sample_index']  # index of example inside its dataset
        meta_mask = input_batch['meta_mask']
        batch_size = images.shape[0]
        scale = input_batch['scale']
        center = input_batch['center']
        bbox_height = input_batch['bbox_height']
        bbox_width = input_batch['bbox_width']
        orig_shape = input_batch['orig_shape']
        focal_length = input_batch['focal']
        bbox_info = input_batch['bbox_info']

        # Feed images in the network to predict camera and SMPL parameters
        if self.options.regressor == 'hmr':
            pred_rotmat, pred_betas, pred_camera = self.model(images)
            # torch.Size([32, 24, 3, 3]) torch.Size([32, 10]) torch.Size([32, 3])
        elif self.options.regressor == 'pymaf_net':
            preds_dict, _ = self.model(images, meta_mask, center, scale, bbox_height, orig_shape, bbox_info,
                                       is_train=True)

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices
        opt_pose, opt_betas = gt_pose, gt_betas

        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3])
        opt_vertices = opt_output.vertices
        opt_sub_vertices = torch.matmul(self.Dmap0, opt_vertices)
        opt_temp_vertices = torch.matmul(self.Dmap1, opt_sub_vertices)

        opt_joints = opt_output.joints

        input_batch['verts'] = opt_vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length,
                                         img_size=[cfg.IMG_RES.WIDTH, cfg.IMG_RES.HEIGHT])

        # get fitted smpl parameters as pseudo ground truth
        valid_fit = has_smpl

        # Render Dense Correspondences
        if self.options.regressor == 'pymaf_net' and cfg.MODEL.PyMAF.AUX_SUPV_ON:
            gt_cam_t_nr = opt_cam_t.detach().clone()
            gt_camera = torch.zeros(gt_cam_t_nr.shape).to(gt_cam_t_nr.device)
            gt_camera[:, 1:] = gt_cam_t_nr[:, :2]
            gt_camera[:, 0] = (2. * self.focal_length / cfg.IMG_RES.HEIGHT) / gt_cam_t_nr[:, 2]
            iuv_image_gt = torch.zeros(
                (batch_size, 3, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE[0], cfg.MODEL.PyMAF.DP_HEATMAP_SIZE[1])).to(self.device)
            depth_image_gt = torch.zeros(
                (batch_size, 1, cfg.MODEL.PyMAF.DP_HEATMAP_SIZE[0], cfg.MODEL.PyMAF.DP_HEATMAP_SIZE[1])).to(self.device)
            if torch.sum(valid_fit.float()) > 0:
                iuv_image_gt[valid_fit] = self.iuv_maker.verts2iuvimg(opt_vertices[valid_fit],
                                                                      cam=gt_camera[valid_fit])  # [B, 3, 56, 56]
                if cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
                    iuv_image_gt = iuv_image_gt[:, :, :, 16:-16]
                if cfg.MODEL.PyMAF.DEPTH_SUPV_ON:
                    depth_image_gt[valid_fit] = self.depth_maker.verts2depthimg(opt_vertices[valid_fit], cam=gt_camera[
                        valid_fit])  # [B, 1, 56, 56]
                    if cfg.MODEL.PyMAF.BACKBONE == 'vitpose':
                        depth_image_gt = depth_image_gt[:, :, :, 16:-16]

            input_batch['iuv_image_gt'] = iuv_image_gt

            uvia_list = iuv_img2map(iuv_image_gt)

        output = preds_dict
        loss_dict = {}

        if self.options.regressor == 'pymaf_net' and cfg.MODEL.PyMAF.AUX_SUPV_ON:
            dp_out = preds_dict['dp_out']
            for i in range(len(dp_out)):
                r_i = i - len(dp_out)

                u_pred, v_pred, index_pred, ann_pred = dp_out[r_i]['predict_u'], dp_out[r_i]['predict_v'], dp_out[r_i][
                    'predict_uv_index'], dp_out[r_i]['predict_ann_index']
                if index_pred.shape[-1] == iuv_image_gt.shape[-1]:
                    uvia_list_i = uvia_list
                else:
                    iuv_image_gt_i = F.interpolate(iuv_image_gt, u_pred.shape[-1], mode='nearest')
                    uvia_list_i = iuv_img2map(iuv_image_gt_i)

                loss_U, loss_V, loss_IndexUV, loss_segAnn = self.body_uv_losses(u_pred, v_pred, index_pred, ann_pred,
                                                                                uvia_list_i, valid_fit)
                loss_dict[f'loss_U{r_i}'] = loss_U
                loss_dict[f'loss_V{r_i}'] = loss_V
                loss_dict[f'loss_IndexUV{r_i}'] = loss_IndexUV
                loss_dict[f'loss_segAnn{r_i}'] = loss_segAnn

        # depth image loss
        if self.options.regressor == 'pymaf_net' and cfg.MODEL.PyMAF.DEPTH_SUPV_ON:
            dpth_out = preds_dict['dpth_out']
            for i in range(len(dpth_out)):
                r_i = i - len(dp_out)
                dpth_pred = dpth_out[r_i]
                loss_Depth = self.depth_losses(dpth_pred, depth_image_gt, valid_fit)
                loss_dict[f'loss_Depth{r_i}'] = loss_Depth

        len_loop = len(preds_dict['smpl_out']) if self.options.regressor == 'pymaf_net' else 1

        # res_scale used to scale loss_keypoints_world
        bbox_width = bbox_width[:, None]
        bbox_height = bbox_height[:, None]
        res_scale = torch.cat((bbox_height, bbox_width), dim=1)
        loss_keypoints_scale = orig_shape / res_scale
        loss_keypoints_scale[:, [0, 1]] = loss_keypoints_scale[:, [1, 0]]
        loss_keypoints_scale = loss_keypoints_scale[:, None, :]

        for l_i in range(len_loop):

            if self.options.regressor == 'pymaf_net':
                if l_i == 0:
                    # initial parameters (mean poses)
                    continue
                pred_rotmat = preds_dict['smpl_out'][l_i]['rotmat']
                pred_betas = preds_dict['smpl_out'][l_i]['theta'][:, 3:13]
                pred_camera = preds_dict['smpl_out'][l_i]['theta'][:, :3]
                pred_keypoints_2d = preds_dict['smpl_out'][l_i]['kp_2d']
                pred_keypoints_2d_world = preds_dict['smpl_out'][l_i]['kp_2d_w']
                pred_joints = preds_dict['smpl_out'][l_i]['kp_3d']
                pred_vertices = preds_dict['smpl_out'][l_i]['verts']
                pred_sub_vertices = preds_dict['smpl_out'][l_i]['sub_verts']
                pred_temp_vertices = preds_dict['smpl_out'][l_i]['temp_verts']
                pred_focal = preds_dict['smpl_out'][l_i]['focal_length']

            # pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
            #                         global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            # pred_vertices = pred_output.vertices
            # pred_joints = pred_output.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t = torch.stack([pred_camera[:, 1],
                                      pred_camera[:, 2],
                                      2 * self.focal_length / (cfg.IMG_RES.HEIGHT * pred_camera[:, 0] + 1e-9)],
                                     dim=-1)

            # Compute loss on SMPL parameters
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
            loss_regr_pose *= cfg.LOSS.POSE_W
            loss_regr_betas *= cfg.LOSS.SHAPE_W
            loss_dict['loss_regr_pose_{}'.format(l_i)] = loss_regr_pose
            loss_dict['loss_regr_betas_{}'.format(l_i)] = loss_regr_betas
            # Compute 2D reprojection loss for the keypoints
            if cfg.LOSS.KP_2D_W > 0:
                loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                                    self.options.openpose_train_weight,
                                                    self.options.gt_train_weight)
                loss_keypoints = loss_keypoints.mean() * cfg.LOSS.KP_2D_W
                loss_dict['loss_keypoints_{}'.format(l_i)] = loss_keypoints

                loss_keypoints_world = self.keypoint_loss(pred_keypoints_2d_world, gt_keypoints_2d_world,
                                                          self.options.openpose_train_weight,
                                                          self.options.gt_train_weight) * loss_keypoints_scale
                loss_keypoints_world = loss_keypoints_world.mean() * cfg.LOSS.KP_2D_W
                # if loss_keypoints_world > 100.:
                #     joblib.dump(pred_keypoints_2d_world,'pred_keypoints_2d_world.pt')
                #     joblib.dump(gt_keypoints_2d_world,'gt_keypoints_2d_world.pt')
                #     joblib.dump(loss_keypoints_scale,'loss_keypoints_scale.pt')
                #     joblib.dump(center,'center.pt')
                #     joblib.dump(orig_shape,'orig_shape.pt')
                #     joblib.dump(bbox_width,'bbox_width.pt')
                #     joblib.dump(bbox_height,'bbox_height.pt')
                #     exit()
                loss_dict['loss_keypoints_world_{}'.format(l_i)] = loss_keypoints_world

            # Compute focal length loss
            if cfg.MODEL.PyMAF.FOCAL_SUPV_ON:
                loss_focal_length = self.criterion_focal(pred_focal, focal_length) * cfg.LOSS.FOCAL_WEIGHTS
                loss_dict['loss_focal_length_{}'.format(l_i)] = loss_focal_length

            # Compute 3D keypoint loss
            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d) * cfg.LOSS.KP_3D_W
            loss_dict['loss_keypoints_3d_{}'.format(l_i)] = loss_keypoints_3d

            # Per-vertex loss for the shape
            if cfg.LOSS.VERT_W > 0 and l_i > 2:
                loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit) * cfg.LOSS.VERT_W
                loss_dict['loss_shape_{}'.format(l_i)] = loss_shape
                loss_shape_sub = self.shape_loss(pred_sub_vertices, opt_sub_vertices, valid_fit) * cfg.LOSS.VERT_W
                loss_dict['loss_shape_sub_{}'.format(l_i)] = loss_shape_sub
                loss_shape_temp = self.shape_loss(pred_temp_vertices, opt_temp_vertices, valid_fit) * cfg.LOSS.VERT_W
                loss_dict['loss_shape_temp_{}'.format(l_i)] = loss_shape_temp

            # Camera
            # force the network to predict positive depth values
            loss_cam = ((torch.exp(-pred_camera[:, 0] * 10)) ** 2).mean()
            loss_dict['loss_cam_{}'.format(l_i)] = loss_cam

        for key in loss_dict:
            if len(loss_dict[key].shape) > 0:
                loss_dict[key] = loss_dict[key][0]

        # print('\n', loss_dict['loss_keypoints_world_3'], '\n')
        # print('\n', loss_dict['loss_keypoints_3d_3'], '\n')
        # print('\n', loss_dict['loss_keypoints_3'], '\n')
        # print('\n', pred_focal.mean(), '\n')

        # print(loss_dict['loss_keypoints_3'])
        # print('\n', loss_dict['loss_keypoints_3'], '\n')
        # loss_dict_new = {}
        # # loss_dict_new['loss_keypoints_world_1'] = loss_dict['loss_keypoints_world_1']
        # # loss_dict_new['loss_keypoints_world_2'] = loss_dict['loss_keypoints_world_2']
        # # loss_dict_new['loss_keypoints_world_3'] = loss_dict['loss_keypoints_world_3']
        # loss_dict_new['loss_focal_length_1'] = loss_dict['loss_focal_length_1']
        # loss_dict_new['loss_focal_length_2'] = loss_dict['loss_focal_length_2']
        # loss_dict_new['loss_focal_length_3'] = loss_dict['loss_focal_length_3']
        # loss_dict = loss_dict_new
        # Compute total loss
        loss = torch.stack(list(loss_dict.values())).sum()

        # print('\n', loss, '\n')
        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output.update({'pred_vertices': pred_vertices.detach(),
                       'opt_vertices': opt_vertices,
                       'pred_cam_t': pred_cam_t.detach(),
                       'opt_cam_t': opt_cam_t})
        loss_dict['loss'] = loss.detach().item()

        if self.step_count % 100 == 0:
            if self.options.multiprocessing_distributed:
                for loss_name, val in loss_dict.items():
                    val = val / self.options.world_size
                    if not torch.is_tensor(val):
                        val = torch.Tensor([val]).to(self.device)
                    dist.all_reduce(val)
                    loss_dict[loss_name] = val
            if self.options.rank == 0:
                for loss_name, val in loss_dict.items():
                    self.summary_writer.add_scalar('losses/{}'.format(loss_name), val, self.step_count)

        return {'preds': output, 'losses': loss_dict}

    def fit(self):
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs,
                          initial=self.epoch_count):
            self.epoch_count = epoch
            self.train(epoch)

            self.validate()

            if self.options.rank == 0:
                performance = self.evaluate()

                # if self.lr_scheduler is not None:
                #     self.lr_scheduler.step(performance)

                # log the learning rate
                for param_group in self.optimizer.param_groups:
                    print(f'Learning rate {param_group["lr"]}')
                    self.summary_writer.add_scalar('lr/model_lr', param_group['lr'], global_step=self.epoch_count)

                is_best = performance < self.best_performance
                if is_best:
                    logger.info('Best performance achived, saved it!')
                    self.best_performance = performance
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0, cfg.TRAIN.BATCH_SIZE,
                                           self.step_count, is_best)

        return

    # @torch.no_grad()
    # def validate(self):
    #     self.model.eval()
    #     start = time.time()
    #     logger.info('Start Validation.')
    #
    #     # initialize
    #     for k, v in self.evaluation_accumulators.items():
    #         self.evaluation_accumulators[k] = []
    #
    #     # Regressor for H36m joints
    #     J_regressor = torch.from_numpy(np.load(path_config.JOINT_REGRESSOR_H36M)).float()
    #
    #     joint_mapper_gt = constants.J45_TO_J24 if self.options.eval_dataset == 'agora' else constants.J24_TO_J14
    #
    #     if self.options.rank == 0:
    #         pbar = tqdm(desc='Eval', total=len(self.valid_ds) // cfg.TEST.BATCH_SIZE)
    #     for i, target in enumerate(self.valid_loader):
    #         if self.options.rank == 0:
    #             pbar.update(1)
    #
    #         # Get GT vertices and model joints
    #         gt_betas = target['betas'].to(self.device)
    #         gt_pose = target['pose'].to(self.device)
    #         gt_pose_clone = gt_pose.clone().reshape(-1, 24, 3)
    #         gt_out = self.smplx(betas=gt_betas.to(torch.float32), body_pose=gt_pose_clone[:, 1:].to(torch.float32),
    #                             global_orient=gt_pose_clone[:, :1].to(torch.float32),
    #                             pos2rot=True)
    #         gt_model_joints = gt_out['joints']
    #         gt_vertices = gt_out['vertices']
    #         target['verts'] = gt_vertices.cpu()
    #         target_j3d = gt_model_joints.cpu()
    #
    #         inp = target['img'].to(self.device, non_blocking=True)
    #         meta_mask = target['meta_mask'].to(self.device, non_blocking=True)
    #         scale = target['scale'].to(self.device, non_blocking=True)
    #         center = target['center'].to(self.device, non_blocking=True)
    #         bbox_height = target['bbox_height'].to(self.device, non_blocking=True)
    #         orig_shape = target['orig_shape'].to(self.device, non_blocking=True)
    #         bbox_info = target['bbox_info'].to(self.device, non_blocking=True)
    #
    #         # J_regressor_batch = J_regressor[None, :].expand(inp.shape[0], -1, -1).contiguous().to(self.device,
    #         #                                                                                       non_blocking=True)
    #
    #         pred_dict, _ = self.model(inp, meta_mask, center, scale, bbox_height, orig_shape, bbox_info, is_train=False,
    #                                   J_regressor=None)
    #         if self.options.rank == 0:
    #             if cfg.TRAIN.VAL_LOOP:
    #                 preds_list = pred_dict['smpl_out']
    #             else:
    #                 preds_list = pred_dict['smpl_out'][-1:]
    #
    #             for preds in preds_list:
    #                 preds['kp_3d'] = preds['smpl_kp_3d']
    #                 preds['verts'] = preds['verts']
    #
    #                 # convert to 14 keypoint format for evaluation
    #                 n_kp = preds['kp_3d'].shape[-2]
    #                 pred_j3d = preds['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
    #                 pred_j3d = pred_j3d[:, joint_mapper_gt, :]
    #                 pred_j3d = pred_j3d - pred_j3d[:, [0], :]
    #
    #                 target_j3d = target_j3d[:, joint_mapper_gt, :]
    #                 target_j3d = target_j3d - target_j3d[:, [0], :]
    #
    #                 pred_verts = preds['verts'].cpu().numpy()
    #                 target_verts = target['verts'].to('cpu').numpy()
    #
    #                 batch_len = target['betas'].shape[0]
    #
    #                 self.evaluation_accumulators['pred_verts'].append(pred_verts)
    #                 self.evaluation_accumulators['target_verts'].append(target_verts)
    #                 self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
    #                 self.evaluation_accumulators['target_j3d'].append(target_j3d)
    #
    #             # if (i + 1) % cfg.VAL_VIS_BATCH_FREQ == 0 and self.options.rank == 0:
    #             #     self.visualize(i, target, 'valid', pred_dict)
    #
    #         del pred_dict, _
    #
    #         batch_time = time.time() - start
    #
    #     if self.options.rank == 0:
    #         pbar.close()

    # for h36m and 3dpw
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        start = time.time()
        logger.info('Start Validation.')

        # initialize
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = []

        # Regressor for H36m joints
        J_regressor = torch.from_numpy(np.load(path_config.JOINT_REGRESSOR_H36M)).float()

        joint_mapper_h36m = constants.H36M_TO_J17 if self.options.eval_dataset == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if self.options.eval_dataset == 'mpi-inf-3dhp' else constants.J24_TO_J14

        if self.options.rank == 0:
            pbar = tqdm(desc='Eval', total=len(self.valid_ds) // cfg.TEST.BATCH_SIZE)
        for i, target in enumerate(self.valid_loader):
            if self.options.rank == 0:
                pbar.update(1)
            inp = target['img'].to(self.device, non_blocking=True)
            J_regressor_batch = J_regressor[None, :].expand(inp.shape[0], -1, -1).contiguous().to(self.device,
                                                                                                  non_blocking=True)
            # Get GT vertices and model joints
            gt_betas = target['betas'].to(self.device)
            gt_pose = target['pose'].to(self.device)
            if 'h36m' in self.options.eval_dataset:
                gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
                gt_vertices = gt_out.vertices
                target['verts'] = gt_vertices
            elif '3dpw' in self.options.eval_dataset:
                # For 3DPW get the 14 common joints from the rendered shape
                gender = target['gender'].to(self.device)
                gt_vertices = self.smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                             betas=gt_betas).vertices
                gt_vertices_female = self.smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                      betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                target_j3d = gt_keypoints_3d - gt_pelvis

                target['target_j3d'] = target_j3d
                target['verts'] = gt_vertices

            inp = target['img'].to(self.device, non_blocking=True)
            meta_mask = target['meta_mask'].to(self.device, non_blocking=True)
            scale = target['scale'].to(self.device, non_blocking=True)
            center = target['center'].to(self.device, non_blocking=True)
            bbox_height = target['bbox_height'].to(self.device, non_blocking=True)
            orig_shape = target['orig_shape'].to(self.device, non_blocking=True)
            bbox_info = target['bbox_info'].to(self.device, non_blocking=True)
            J_regressor_batch = J_regressor[None, :].expand(inp.shape[0], -1, -1).contiguous().to(self.device,
                                                                                                  non_blocking=True)

            pred_dict, _ = self.model(inp, meta_mask, center, scale, bbox_height, orig_shape, bbox_info, is_train=False,
                                      J_regressor=J_regressor_batch)

            if self.options.rank == 0:
                if cfg.TRAIN.VAL_LOOP:
                    preds_list = pred_dict['smpl_out']
                else:
                    preds_list = pred_dict['smpl_out'][-1:]

                for preds in preds_list:
                    # convert to 14 keypoint format for evaluation
                    n_kp = preds['kp_3d'].shape[-2]
                    pred_j3d = preds['kp_3d'].view(-1, n_kp, 3).cpu().numpy()

                    if 'h36m' in self.options.eval_dataset:
                        target_j3d = target['pose_3d'].cpu()
                        target_j3d = target_j3d[:, joint_mapper_gt, :-1]
                    elif '3dpw' in self.options.eval_dataset:
                        target_j3d = target['target_j3d'].cpu()

                    pred_verts = preds['verts'].cpu().numpy()
                    target_verts = target['verts'].to('cpu')
                    target_verts = target_verts.numpy()

                    batch_len = target['betas'].shape[0]

                    self.evaluation_accumulators['pred_verts'].append(pred_verts)
                    self.evaluation_accumulators['target_verts'].append(target_verts)
                    self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                    self.evaluation_accumulators['target_j3d'].append(target_j3d)

                # if (i + 1) % cfg.VAL_VIS_BATCH_FREQ == 0 and self.options.rank == 0:
                #     self.visualize(i, target, 'valid', pred_dict)

            del pred_dict, _

            batch_time = time.time() - start

        if self.options.rank == 0:
            pbar.close()

    def evaluate(self):
        if cfg.TRAIN.VAL_LOOP:
            step = cfg.MODEL.PyMAF.N_ITER + 1
        else:
            step = 1

        num_poses = len(self.evaluation_accumulators['pred_j3d']) * cfg.TRAIN.BATCH_SIZE // step
        print(f'Evaluating on {num_poses} number of poses ...')

        for loop_id in range(step):
            pred_j3ds = self.evaluation_accumulators['pred_j3d'][loop_id::step]
            pred_j3ds = np.vstack(pred_j3ds)
            pred_j3ds = torch.from_numpy(pred_j3ds).float()

            target_j3ds = self.evaluation_accumulators['target_j3d'][loop_id::step]
            target_j3ds = np.vstack(target_j3ds)
            target_j3ds = torch.from_numpy(target_j3ds).float()

            # Absolute error (MPJPE)
            errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            S1_hat = compute_similarity_transform_batch(pred_j3ds.numpy(), target_j3ds.numpy())
            S1_hat = torch.from_numpy(S1_hat).float()
            errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            pred_verts = self.evaluation_accumulators['pred_verts'][loop_id::step]
            pred_verts = np.vstack(pred_verts)
            pred_verts = torch.from_numpy(pred_verts).float()

            target_verts = self.evaluation_accumulators['target_verts'][loop_id::step]
            target_verts = np.vstack(target_verts)
            target_verts = torch.from_numpy(target_verts).float()
            errors_pve = torch.sqrt(((pred_verts - target_verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            m2mm = 1000
            pve = np.mean(errors_pve) * m2mm
            mpjpe = np.mean(errors) * m2mm
            pa_mpjpe = np.mean(errors_pa) * m2mm

            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'pve': pve,
            }

            loop_id -= step  # to ensure the index of latest prediction is always -1
            log_str = f'Epoch {self.epoch_count}, step {loop_id}  '
            log_str += ' '.join([f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
            logger.info(log_str)

            for k, v in eval_dict.items():
                self.summary_writer.add_scalar(f'eval_error/{k}_{loop_id}', v, global_step=int(self.epoch_count))

        # empty accumulators
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k].clear()

        return mpjpe
