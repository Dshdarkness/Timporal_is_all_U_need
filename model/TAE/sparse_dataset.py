import torch
import os
import numpy as np
from numpy.random import randint
import torch.utils.data as data
from PIL import Image
import glob
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize
from aug import IsotropicResize
from torch.utils.data import DataLoader, Dataset

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
                #IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                #IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                #IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
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


class Deepfakes(data.Dataset):
    def __init__(self, dataset_root, num_segments=8, new_length=1, mode='train',dataAug=False, size=320, dataset_name=None, image_tmpl='{:03d}.png', frame_num=100):
        self.dataset_root = dataset_root
        self.num_segments = num_segments
        self.new_length = new_length
        self.mode = mode
        self.dataAug = dataAug
        self.size = size
        self.dataset_name = dataset_name
        self.image_tmpl = image_tmpl
        self.data_list = self.collect_image(self.dataset_root)
        self.frame_num = frame_num


    def collect_image(self, root):
        all_videos_list= []
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

        return all_videos_list


    def processing_img(self, image, size):
        if self.mode != 'train':
            image = cv2.resize(image, (size, size)) # reseize 320*320 -> size*size
        #if self.dataAug:  # only get aug for real video
        #    image = get_aug(image)
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image        

    def load_img(self,video_path, idx):
        img = Image.open(os.path.join(video_path, self.image_tmpl.format(idx))).convert('RGB')
        img = np.array(img)
        return img
    # 全局稀疏采样函数 输入一个视频的帧数(这里每个视频都只有100帧)， 输出是列表(假如分十段)，如[5,14,23,36,48,53.....]


    def sample_indices(self, frame_num):
    
        average_duration = (frame_num - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif frame_num > self.num_segments:
            offsets = np.sort(randint(frame_num - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1            
    

    def __getitem__(self, index):
        video_path = self.data_list[index]
        frame_num = len(os.listdir(video_path))
        segment_indices = self.sample_indices(frame_num)
        images = list()
        for seg_ind in segment_indices:
            p = int(seg_ind)

            for i in range(self.new_length):
                seg_imgs = self.load_img(video_path, p)
                seg_imgs = self.processing_img(seg_imgs, self.size) # tensor [3, size, size]
                images.extend([seg_imgs])
                if p < frame_num:
                    p += 1
        images = torch.cat(images, dim=0) # tensor [3*segments, size, size]

        labels = label_dic[video_path.split('/')[-2]]
        sample = (images, labels)
        return sample

    def __len__(self):
        return len(self.data_list)



if __name__ == '__main__':
    train_set = Deepfakes(dataset_root='/root/autodl-tmp/FF_data/FF_LQ/train_face', dataset_name='Deepfakes', mode='train')
    train_dl = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
    val_set = Deepfakes(dataset_root='/root/autodl-tmp/FF_data/FF_LQ/train_face', dataset_name='Deepfakes', mode='val')
    val_dl = DataLoader(val_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
    for i,data in enumerate(train_dl):
        img,label = data
        print(img.shape) #[8,24,320，320]
        break

    for i,data in enumerate(val_dl):
        img,label = data
        print(img.shape) #[8,24,320,320]
        break