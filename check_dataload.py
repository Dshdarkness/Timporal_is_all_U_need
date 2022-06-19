# -*-coding:utf-8-*-
import os
import math
import time
from torch.utils.data import DataLoader

from dataset import Deepfakes
# from video_dataset import Deepfakes
from utils import *  # including other libraries like torch, tqdm, etc.


def main():
    global best_model_path
    dataset_path = '/root/autodl-tmp/FF_data/FF_LQ/train_face'
    assert os.path.exists(dataset_path), "dataset_path is not exists, plz try again."

    # set gpu
    device = "cpu"

    # params setting
    h_params = {
        'BATCH_SIZE': 64,
        'MAX_EPOCH': 30,
        'NUM_WORKERS': 4,
        'NUM_CLASSES': 2,
        'DEVICE': device}
    # print(h_params)

    # obtain dataset
    # dataset_name = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    dataset_name = ['Deepfakes']

    for name in dataset_name:
        print(f"Now, start check training {name} dataset")
        # create a folder to save model

        # init dataset
        train_dataset = Deepfakes(dataset_root=dataset_path, dataset_name=name, size=320,
                                frame_num=100, mode='train', dataAug=True)
        val_dataset = Deepfakes(dataset_root=dataset_path, dataset_name=name, size=320,
                                frame_num=100, mode='val', dataAug=False)
        test_dataset = Deepfakes(dataset_root=dataset_path, dataset_name=name, size=320,
                                frame_num=100, mode='test', dataAug=False)

        print(f"Train Data: {len(train_dataset)}")
        print(f"Validation Data: {len(val_dataset)}")
        print(f"Test Data: {len(test_dataset)}")

        # dataloader
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=h_params['BATCH_SIZE'],
                                  pin_memory=True,
                                  num_workers=h_params['NUM_WORKERS'],
                                  shuffle=True,
                                  drop_last=True,)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=h_params['BATCH_SIZE'],
                                pin_memory=True,
                                num_workers=h_params['NUM_WORKERS'],
                                shuffle=False,
                                drop_last=True,)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                pin_memory=True,
                                num_workers=h_params['NUM_WORKERS'],
                                shuffle=False,)

        for _, train_data in enumerate(train_loader):
                train_images, train_labels = train_data        
        for _, val_data in enumerate(val_loader):
                val_images, val_labels = val_data        
        for _, test_data in enumerate(test_loader):
                test_images, test_labels = test_data



        print(f"------->>>>>>Finished checking<<<<<<<--------, "
              f"dataset_name= {name}")

if __name__ == '__main__':
    main()
