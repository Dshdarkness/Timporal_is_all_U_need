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

label_dic = {'fake': 0, 'real': 1}


def get_aug(img_arr):
    trans = A.Compose([
        A.OneOf([
            A.RandomGamma(gamma_limit=(60, 120), p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
            A.GaussianBlur(),
        ]),
        # A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        # A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR, p=0.3),
        # A.HorizontalFlip(p=0.5),
        # A.OneOf([
        #     A.CoarseDropout(),
        #     A.GridDistortion(),
        #     A.GridDropout(),
        #     A.OpticalDistortion()
        # ]),
    ])
    trans_img = trans(image=img_arr)['image']
    return trans_img


class Deepfakes(Dataset):
    '''
    pipeline: given an index from get_item --> video --> frames --> stack frames --> return (frames_stack, label)
    '''
    def __init__(self, dataset_root=None, dataset_name=None, frame_num=300, size=224, mode=None, dataAug=False):
        self.data_root = dataset_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.dataAug = dataAug
        self.frame_num = frame_num
        self.data_list = self.collect_video(self.data_root)
        self.size = size

    def processing_img(self, image, size, seed):
        if self.mode != 'train':
            image = cv2.resize(image, (size, size))
        if self.dataAug:
            random.seed(seed)  # important !!! to ensure the aug seed for each frame is the same!!!
            image = get_aug(image)
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image

    def collect_video(self, root):
        all_videos_list = []
        # e.g. ffdataset, Deepfakes, fake, video_0
        if self.mode == 'train':
            if ',' not in self.dataset_name:  # train one dataset
                all_videos_list = glob.glob(os.path.join(root, 'train', self.dataset_name, '*', '*'))
            else:  # train four dataset together, Deepfakes,Face2Face
                dataset_list = self.dataset_name.split(',')
                for dl in dataset_list:
                    all_videos_list.extend(glob.glob(os.path.join(root, 'train', dl, 'fake', '*')))
                # real video
                all_videos_list.extend(glob.glob(os.path.join(root, 'train', dl, 'real', '*')) * 4)
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

        return all_videos_list * 4

    def stack_imgs(self, video_path: str):
        # *** get shuffle, sorted frames *** #
        img_list = glob.glob(os.path.join(video_path, '*'))
        random.shuffle(img_list)
        frames = img_list[:self.frame_num]
        frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # *** processing img frame by frame *** #
        seed = random.randint(0,99999)
        X = []
        for frame in frames:
            image = cv2.imread(frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.processing_img(image, self.size, seed)
            X.append(image)
        img_stack = torch.stack(X, dim=0)
        return img_stack

    def __getitem__(self, index):
        video_path = self.data_list[index]
        label = label_dic[video_path.split('/')[-2]]
        img_stack = self.stack_imgs(video_path)
        sample = (img_stack, label)
        return sample

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset_path = '/root/autodl-tmp/FF_data/FF_HQ/train_face'
    name = 'Deepfakes'
    train_dataset = Deepfakes(dataset_root=dataset_path, dataset_name=name, size=320,
                              frame_num=8, mode='train', dataAug=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=30,
                              pin_memory=False,
                              num_workers=0,
                              shuffle=True)
    for i, data in enumerate(train_loader):
        img, label = data
