import os.path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import cv2

from utils.imutils import crop

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


class TCMRDataset():
    def __init__(self, is_train=True):
        self.db = self.load_db()
        self.is_train = is_train
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db = joblib.load('data/3dpw_test_db.pt')
        return db

    def process_image(self, img_file, input_res=224):
        """Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """
        normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)
        if self.datasetname == 'insta':
            img = cv2.imread(img_file)[:, :, ::-1].copy()
        else:
            img = joblib.load(img_file)
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200

        img_np = crop(img, center, scale, (input_res, input_res))
        img = img_np.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        norm_img = normalize_img(img.clone())
        return norm_img

    def get_single_item(self, index):
        path = self.db[index]
        img = self.process_image(path)

        return img