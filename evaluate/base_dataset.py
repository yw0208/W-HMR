# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/base_dataset.py

from __future__ import division

import cv2
import joblib
import torch
import random
import numpy as np
from os.path import join
import albumentations as A

from pare.dataset.coco_occlusion import load_coco_occluders, load_pascal_occluders, occlude_with_coco_objects, \
    occlude_with_pascal_objects
from pare.utils.image_utils import random_crop
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from core import path_config, constants
from core.cfgs import cfg
from core.path_config import PASCAL_ROOT
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, transform_pts, rot_aa
from pare.models import SMPL

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/path_config.py.
    """

    def __init__(self, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super().__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.img_dir = path_config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self.data = np.load(path_config.DATASET_FILES[is_train][dataset], allow_pickle=True)

        self.imgname = self.data['imgname']

        self.dataset_dict = {dataset: 0}

        logger.info('len of {}: {}'.format(self.dataset, len(self.imgname)))

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)  # (N, 72)
            self.betas = self.data['shape'].astype(np.float)  # (N, 10)

            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname), dtype=np.float32)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)

        # Get SMPL 2D keypoints
        try:
            self.smpl_2dkps = self.data['smpl_2dkps']
            self.has_smpl_2dkps = 1
        except KeyError:
            self.has_smpl_2dkps = 0

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        # Get scores
        self.score = self.data['score']

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling

        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor] but it is zero with probability 3/5
            if np.random.uniform() > 0.6:
                rot = min(2 * self.options.rot_factor,
                          max(-2 * self.options.rot_factor, np.random.randn() * self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.scale_factor,
                     max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))

            if np.random.uniform() > 0.7:
                t_r_w = np.random.randn()
                t_r_h = np.random.randn()

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, [cfg.IMG_RES.WIDTH, cfg.IMG_RES.HEIGHT], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)

        # add Color Jitter
        if self.is_train:
            albumentation_aug = A.Compose(transforms=[A.ColorJitter(brightness=(0.2, 0.4), contrast=(0.3, 0.5), p=0.2)])
            rgb_img = albumentation_aug(image=rgb_img)['image']

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_smpl=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [cfg.IMG_RES.WIDTH, cfg.IMG_RES.HEIGHT], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / np.array([cfg.IMG_RES.WIDTH, cfg.IMG_RES.HEIGHT]) - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp, is_smpl)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f, is_smpl=False):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S, is_smpl)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_crop_shape(self, center, scale, res, rot=0):
        # Upper left point
        ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(transform([res[0] + 1,
                                 res[1] + 1], center, scale, res, invert=1)) - 1

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad

        return ul, br

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        center_orig = center.copy()
        scale_orig = scale.copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        scale = sc * scale

        # apply crop augmentation
        # if self.is_train and np.random.rand() < self.options.CROP_PROB:
        #     crop_scale_factor = np.random.rand() * 0.3 + 0.5
        #     center, scale = random_crop(center, scale, crop_scale_factor=crop_scale_factor, axis='y')

        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)
            orig_shape = np.array(img.shape)[:2]
        except:
            logger.error('fail while loading {}'.format(imgname))

        kp_is_smpl = True if self.dataset == 'surreal' else False

        # Process image
        try:
            img = self.rgb_processing(img, center, scale, rot, flip, pn)
        except:
            center = center_orig
            scale = scale_orig
            img = self.rgb_processing(img, center, scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['img'] = item['img'][:, :, 32:-32]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            pose = self.pose_processing(pose, rot, flip)
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 2D SMPL joints
        if self.has_smpl_2dkps:
            smpl_2dkps = self.smpl_2dkps[index].copy()
            smpl_2dkps = self.j2d_processing(smpl_2dkps, center, scale, rot, f=0)
            smpl_2dkps[smpl_2dkps[:, 2] == 0] = 0
            if flip:
                smpl_2dkps = smpl_2dkps[constants.SMPL_JOINTS_FLIP_PERM]
                smpl_2dkps[:, 0] = - smpl_2dkps[:, 0]
            item['smpl_2dkps'] = torch.from_numpy(smpl_2dkps).float()
        else:
            item['smpl_2dkps'] = torch.zeros(24, 3, dtype=torch.float32)

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip, kp_is_smpl)).float()
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(
            self.j2d_processing(keypoints, center, scale, rot, flip, kp_is_smpl)).float()

        #
        ul, br = self.get_crop_shape(center, scale, [cfg.IMG_RES.WIDTH, cfg.IMG_RES.HEIGHT], rot)
        bbox_res = torch.Tensor([br[0] - ul[0], br[1] - ul[1]])

        # create mask
        mvm_percent = 0.3
        mvm_mask = np.ones((431, 1))
        if self.is_train:
            num_vertices = 431
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices)  # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices), replace=False, size=masked_num)
            mvm_mask[indices, :] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        item['meta_mask'] = mvm_mask
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(scale)
        item['center'] = (ul + br) / 2.
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        item['score'] = self.score[index]
        item['imgPath'] = imgname.split('/')[-1]

        item['bbox_height'] = bbox_res[1]
        item['bbox_width'] = bbox_res[0]
        item['focal'] = np.sqrt((np.power(orig_shape[0], 2) + np.power(orig_shape[1], 2))).astype(np.float32)
        img_center = orig_shape[[1, 0]] / 2.
        cx, cy = center[0:1] - img_center[0:1], center[1:] - img_center[1:]
        item['bbox_info'] = (
                    np.concatenate((cx, cy, bbox_res[1:], orig_shape[1:], orig_shape[0:1])) / item['focal']).astype(
            np.float32)

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)
