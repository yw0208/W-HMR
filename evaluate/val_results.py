import argparse
import os
import pickle
import shutil

import numpy as np
import smplx
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.append('.')
from core.cfgs import parse_args
from core import path_config
from evaluate.base_dataset import BaseDataset
from models import whmr_net
from utils.geometry import perspective_projection


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]

    tz = 2 * focal_length / (bbox_height * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t


if __name__ == '__main__':
    treshold = 0.75
    data = 'agora_test'
    checkpoint_file='logs/pymaf_vitpose/pymaf_vitpose_as_lp3_mlp256-128-64-32_Oct21-18-33-38-hig/checkpoints/model_best.pt'
    # delete past results
    if data == 'agora':
        folder_path = '/opt/data/private/others/val_results'
    else:
        folder_path = '/opt/data/private/others/test_results'

    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    model_gt = smplx.create('/opt/data/private/projects/PyMAF-smpl/data', model_type='smpl', gender='neutral',
                            ext='npz').to('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml', help='config file path for PyMAF.')
    parser.add_argument('--misc', default=None, type=str, nargs="*", help='other parameters')
    args = parser.parse_args()
    parse_args(args)
    mymodel = whmr_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to('cuda')
    checkpoint = torch.load(checkpoint_file)
    mymodel.load_state_dict(checkpoint['model'], strict=True)
    mymodel.eval()

    dataset = BaseDataset(dataset=data, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, prefetch_factor=8, shuffle=False)

    count = -1
    last_imgname = 'ha'
    pbar = tqdm(total=len(dataset) // 1)
    with torch.no_grad():
        for i, target in enumerate(dataloader):
            pbar.update(1)

            if target['score'] < treshold:
                continue

            try:
                inp = target['img'].to('cuda', non_blocking=True)
                meta_mask = target['meta_mask'].to('cuda', non_blocking=True)
                center=target['center'].to('cuda', non_blocking=True)
                scale=target['scale'].to('cuda', non_blocking=True)
                bbox_height=target['bbox_height'].to('cuda', non_blocking=True)
                orig_shape=target['orig_shape'].to('cuda', non_blocking=True)
                bbox_info=target['bbox_info'].to('cuda', non_blocking=True)
            except:
                continue

            imgname = target['imgPath'][0].split('.')[0]
            if imgname != last_imgname:
                last_imgname = imgname
                count = 1
            else:
                count += 1
            resul_name = imgname + '_personId_{}'.format(count) + '.pkl'
            if data == 'agora':
                new_file_list = resul_name.split('_')
                new_file_list = new_file_list[:-3] + new_file_list[-2:]
                resul_name = '_'.join(new_file_list)
                resul_name = os.path.join('/opt/data/private/others/val_results', resul_name)
            else:
                resul_name = os.path.join('/opt/data/private/others/test_results', resul_name)
            # if os.path.exists(resul_name):
            #     continue

            pred_dict, _ = mymodel(inp, meta_mask,center, scale, bbox_height, orig_shape,bbox_info, is_train=False, J_regressor=None)
            preds_list = pred_dict['smpl_out'][-1:]
            preds = preds_list[0]
            # print(preds['verts'][0])
            # exit()

            preds['kp_3d'] = preds['smpl_kp_3d']
            preds['verts'] = preds['verts']
            pred_camera = preds['pred_cam'].to('cuda')
            focal_length=preds['focal_length'][0]
            # focal_length=5000.

            if data == 'agora':
                pred_cam_t = convert_pare_to_full_img_cam(pred_camera, target['bbox_height'].to('cuda'),
                                                          target['center'].to('cuda'), 1280, 720, focal_length)
            else:
                pred_cam_t = convert_pare_to_full_img_cam(pred_camera, target['bbox_height'].to('cuda'),
                                                          target['center'].to('cuda'), 1280, 720, focal_length)
            if data == 'agora':
                camera_center = torch.tensor([[640, 360]]).cuda()
            else:
                camera_center = torch.tensor([[640, 360]]).cuda()

            pred_keypoints_2d = perspective_projection(preds['kp_3d'],
                                                       rotation=torch.eye(3, device='cuda').unsqueeze(0).expand(1, -1,
                                                                                                                -1),
                                                       translation=pred_cam_t,
                                                       focal_length=focal_length,
                                                       camera_center=camera_center)

            if data == 'agora':
                result_joints = {
                    'joints': np.array(pred_keypoints_2d[0][:24].detach().cpu()) * 3.,
                    'verts': np.array(preds['verts'].squeeze(0).detach().cpu()),
                    'allSmplJoints3d': np.array(preds['smpl_kp_3d'].squeeze(0).detach().cpu())[:24]
                }
            else:
                result_joints = {
                    'joints': np.array(pred_keypoints_2d[0][:24].detach().cpu())*3,
                    'verts': np.array(preds['verts'].squeeze(0).detach().cpu()),
                    'allSmplJoints3d': np.array(preds['smpl_kp_3d'].squeeze(0).detach().cpu())[:24]
                }
            with open(resul_name, 'wb') as handle:
                pickle.dump(result_joints, handle)
