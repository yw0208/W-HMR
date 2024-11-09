# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from loguru import logger
from torchvision.transforms import transforms
from multi_person_tracker import MPT
from pare.utils.vibe_image_utils import get_single_image_crop_demo
import sys

from utils.renderer_cam import render_image_group

sys.path.append('.')


from core import path_config
from models import whmr_net

MIN_NUM_FRAMES = 0


class SPECTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model()
        self._load_pretrained_model()
        self.model.eval()
        self.data_transform = transforms.Compose([
            transforms.Resize(600),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        # ========= Define SPEC model ========= #

        model = whmr_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.args.ckpt}')
        ckpt = torch.load(self.args.ckpt)['model']
        self.model.load_state_dict(ckpt, strict=True)
        logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_detector(self, image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=self.args.display,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = mot.detect(image_folder)
        return bboxes

    def run_camcalib(self, image_folder, output_folder):
        cmd = f'python scripts/camcalib_demo.py --img_folder {image_folder} --out_folder {output_folder}/camcalib --no_save'
        os.system(cmd)

    @torch.no_grad()
    def run_on_image_folder(self, image_folder, detections, output_path, output_img_folder, bbox_scale=1.0):
        image_file_names = [
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ]
        image_file_names = sorted(image_file_names)

        for img_idx, img_fname in enumerate(image_file_names):
            dets = detections[img_idx]

            if len(dets) < 1:
                continue

            img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            pil_img = Image.open(img_fname).convert('RGB')
            norm_img = self.data_transform(pil_img)
            full_img = norm_img
            orig_height, orig_width = img.shape[:2]

            inp_images = torch.zeros(len(dets), 3, 256,
                                     256, device=self.device, dtype=torch.float)

            batch_size = inp_images.shape[0]

            bbox_scale = []
            bbox_center = []
            for det_idx, det in enumerate(dets):
                bbox = det
                norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                    img,
                    bbox,
                    kp_2d=None,
                    scale=1.0,
                    crop_size=256
                )
                inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_scale.append(bbox[2] / 200.)
                bbox_center.append([bbox[0], bbox[1]])

            bbox_center = torch.tensor(bbox_center)
            bbox_center_np = np.array(bbox_center)
            bbox_scale = torch.tensor(bbox_scale)
            bbox_scale_np = np.array(bbox_scale.unsqueeze(1))
            img_h = torch.tensor(orig_height).repeat(batch_size)
            img_w = torch.tensor(orig_width).repeat(batch_size)
            orig_shape = torch.cat([img_h.unsqueeze(1), img_w.unsqueeze(1)], dim=1)
            orig_shape_np = np.array(orig_shape)

            pesudo_focal = np.sqrt((np.power(orig_shape_np[:, 0], 2) + np.power(orig_shape_np[:, 1], 2))).astype(
                np.float32)
            pesudo_focal=pesudo_focal[:,None]
            img_center = orig_shape_np[:, [1, 0]] / 2.
            cx, cy = bbox_center_np[:, 0:1] - img_center[:, 0:1], bbox_center_np[:, 1:] - img_center[:, 1:]
            bbox_info = (
                    np.concatenate((cx, cy, 200 * bbox_scale_np, orig_shape_np[:, 1:], orig_shape_np[:, 0:1]),
                                   axis=1) / pesudo_focal).astype(
                np.float32)
            bbox_info = torch.tensor(bbox_info)

            # cam_rotmat, cam_intrinsics, cam_vfov, cam_pitch, cam_roll, cam_focal_length = \
            #     read_cam_params(output_path, img_fname, (orig_height, orig_width))
            # print(cam_pitch,cam_roll,cam_vfov)
            # import IPython; IPython.embed(); exit()
            output = self.model(
                x=inp_images[:, :, :, 32:-32],
                meta_masks=None,
                center=bbox_center.float().to(self.device),
                scale=bbox_scale.float().to(self.device),
                bbox_height=(200 * bbox_scale).float().to(self.device),
                orig_shape=orig_shape.float().to(self.device),
                bbox_info=bbox_info.float().to(self.device),
                is_train=False,
                J_regressor=None,
                full_x=full_img.unsqueeze(0).repeat(batch_size, 1, 1,1).float().to(self.device),
            )

            for k, v in output.items():
                output[k] = v.cpu().numpy()

            del inp_images

            if not self.args.no_save:
                save_f = os.path.join(
                    output_path, 'whmr_results',
                    os.path.basename(img_fname).replace(img_fname.split('.')[-1], 'pkl')
                )
                joblib.dump(output, save_f)

            if not self.args.no_render:
                local_pred_vertices = torch.from_numpy(output['local_smpl_vertices'])
                pred_vertices = torch.from_numpy(output['smpl_vertices'])
                pred_cam_t = torch.from_numpy(output['pred_cam_t'])
                cam_focal_length=output['focal_length']
                render_rotmat = output['render_rotmat']  # pyrender opengl convention



                cy, cx = orig_height // 2, orig_width // 2

                # import IPython; IPython.embed(); exit()



                for i in range(batch_size):
                    focal_length = (cam_focal_length[i], cam_focal_length[i])
                    cam_params = None
                    render_rotmat = np.eye(3)
                    if self.args.save_obj:
                        mesh_folder = os.path.join(output_path, 'meshes', os.path.basename(img_fname).split('.')[0])
                        os.makedirs(mesh_folder, exist_ok=True)
                        mesh_filename = os.path.join(mesh_folder, f'{i:06d}.obj')

                    fname, img_ext = os.path.splitext(img_fname)
                    save_filename = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}{img_ext}')

                    render_img = render_image_group(
                        image=img,
                        camera_translation=pred_cam_t[i],
                        vertices=pred_vertices[i],
                        local_vertices=local_pred_vertices[i],
                        camera_rotation=torch.from_numpy(render_rotmat),
                        focal_length=focal_length,
                        camera_center=(cx, cy),
                        save_filename=save_filename,
                        cam_params=cam_params,
                        mesh_filename=mesh_filename,
                    )

                    if self.args.display:
                        cv2.imshow('W-HMR results', img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        if self.args.display:
            cv2.destroyAllWindows()
