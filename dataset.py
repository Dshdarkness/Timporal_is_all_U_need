import os
import cv2
import pickle
import glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import random
import torch
import os
import PIL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize
from aug import IsotropicResize

label_dic = {'fake': 0, 'real': 1}

def get_aug(img_arr):
    # trans = A.Compose([
    #         A.OneOf([
    #             A.RandomGamma(gamma_limit=(60, 120), p=0.9),
    #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    #             A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
    #             A.GaussianBlur(),
    #         ]),
    #         A.HorizontalFlip(p=0.5),
    #         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
    #                             interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
    #         A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5),
    #         A.OneOf([
    #                 A.CoarseDropout(),
    #                 A.GridDistortion(),
    #                 A.GridDropout(),
    #                 A.OpticalDistortion()
    #                 ]),
    # ])
    size = img_arr.shape[0]
    trans = A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.3),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.HorizontalFlip(),
            A.OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            
            A.OneOf([
                    A.CoarseDropout(),
                    A.GridDistortion(),
                    A.GridDropout(),
                    A.OpticalDistortion()
                    ]),
    ]
    )
    trans_img = trans(image=img_arr)['image']
    return trans_img

class Deepfakes(Dataset):
    def __init__(self, dataset_root=None, dataset_name=None, frame_num=300, size=224, mode=None, dataAug=False):
        self.data_root = dataset_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.dataAug = dataAug
        self.frame_num = frame_num
        self.data_list = self.collect_image(self.data_root)
        self.size = size

    def processing_img(self, image, size, label):
        if self.mode != 'train':
            image = cv2.resize(image, (size, size)) # reseize 320*320 -> size*size
        if self.dataAug:  # only get aug for real video
            image = get_aug(image)
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image

    def collect_image(self, root):
        image_path_list, all_videos_list = [], []
        # e.g. ffdataset, Deepfakes, fake, video_0
        if self.mode == 'train':
            if ',' not in self.dataset_name:  # train one dataset
                all_videos_list = glob.glob(os.path.join(root, 'train', self.dataset_name, '*', '*'))
            else:  # train four dataset together, Deepfakes,Face2Face
                dataset_list = self.dataset_name.split(',')
                for dl in dataset_list:
                    all_videos_list.extend(glob.glob(os.path.join(root, 'train', dl, 'fake', '*')))
                # real video
                all_videos_list.extend(glob.glob(os.path.join(root, 'train', dl, 'real', '*')))
        elif self.mode == 'val':
            if ',' not in self.dataset_name:
                all_videos_list = glob.glob(os.path.join(root, 'val', self.dataset_name, '*', '*'))
            else:
                dataset_list = self.dataset_name.split(',')
                for dl in dataset_list:
                    all_videos_list.extend(glob.glob(os.path.join(root, 'val', dl, 'fake', '*')))
                # real video
                all_videos_list.extend(glob.glob(os.path.join(root, 'val', dl, 'real', '*')))
        elif self.mode == 'test':
            if ',' not in self.dataset_name:
                all_videos_list = glob.glob(os.path.join(root, 'test', self.dataset_name, '*', '*'))
            else:
                dataset_list = self.dataset_name.split(',')
                for dl in dataset_list:
                    all_videos_list.extend(glob.glob(os.path.join(root, 'test', dl, 'fake', '*')))
                # real video
                all_videos_list.extend(glob.glob(os.path.join(root, 'test', dl, 'real', '*')))
        else:
            raise ValueError("dataset_root error, please check the dataset path!")

        # split is one of the four different ffdataset
        for video in all_videos_list:
            img_list = os.listdir(video)
            random.shuffle(img_list)
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(video, img)
                image_path_list.append(img_path)
        return image_path_list

    def __getitem__(self, index):
        image_path = self.data_list[index]
        label = label_dic[image_path.split('/')[-3]]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to rgb or ycr_cb
        img = self.processing_img(img, self.size, label)
        sample = (img, label)
        return sample

    def __len__(self):
        return len(self.data_list)
