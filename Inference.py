import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import transforms
import PIL.Image as Image
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
from model.CIPUCNet import CIPUCNet
from dataset import Deepfakes

global embeddings
label_dic = {'fake': 0, 'real': 1}

def collate_fn(x):
    return x[0]

def load_checkpoint(filepath):
    print("start loading checkpoint.....")
    checkpoint = torch.load(filepath)
    #print(checkpoint)
    model = checkpoint['model']
    # print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def deepfake_detection(test_loader):
    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for step, test_data in enumerate(test_loader):
            test_images = test_data
            outputs = net(test_images.to(device))
            pred_classes = torch.max(outputs, dim=1)[1]
            result_lis = [pc for pc in pred_classes.to("cpu").numpy()]
            pred_classes = np.array([0]) if result_lis.count(0) > result_lis.count(1) else np.array([1])
            fake_prob = torch.max(torch.softmax(outputs, dim=1), dim=1)[0]
            fake_prob = fake_prob.to("cpu").numpy()[0]     # prob(float)
    return pred_classes, fake_prob

# params
base_root = os.getcwd()
model_path = 'finalModel.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
workers = 0 if os.name == 'nt' else 4

# face recognition
mtcnn = MTCNN(keep_all=True,
              image_size=224,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], factor=0.709,
              device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# deepfake detection
frames_need_to_detect = 199
net = load_checkpoint(model_path)
net.to(device)
net.device_ids = [0]

# task_2
task_2 = False
score_yuzhi = 0.95  # a boundary condition for face recognition
if task_2:
    test_images_path = os.path.join(base_root, 'test_images')
    dataset = datasets.ImageFolder(test_images_path)
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    aligned = []
    for x, y in loader:
        # x is image and y is a int(class)
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
    aligned = torch.stack(aligned).to(device)    # 6,3,224,224; 6 represent the total num of images
    embeddings = resnet(aligned).detach().cpu()  # 6, 512

# videos
videos_root = "test_videos"
labels = ['fake', 'real', ]
task = 'task2' if task_2 else 'task1'
#task = 'test'
start_video_frame = 100

y_pred, y_true = [], []
for label in labels:
    # get videos respectively from 'real' dir and 'fake' dir
    video_path = os.path.join(base_root, videos_root, task, label)
    all_videos = os.listdir(video_path)
    all_videos = sorted(all_videos)

    # get label
    label_value = np.array([label_dic[label]])  # e.g. array[0]

    # get video from videos dir
    for video in all_videos:
        per_video_path = os.path.join(video_path, video)
        assert os.path.exists(per_video_path), f'Video path is not exist! {per_video_path}'
        cap = cv2.VideoCapture(per_video_path)
        i = 1
        new_faces = []
        while cap.isOpened():
            if i != 1 and i - start_video_frame == frames_need_to_detect:
                break
            ret, frame = cap.read()
            if not ret:
                break
            # assert ret, f'Cannot read video, some problem occur! {per_video_path}'
            i += 1
            #cv2.imshow('e', frame)
            #if cv2.waitKey(1) == ord('q'):
            #    break

            # we dont need the frames from 0 -> start_video_frame
            if i < start_video_frame:
                continue
            frame = Image.fromarray(frame)
            # do face detection
            faces, prob = mtcnn(frame, return_prob=True)
            if faces is None:
                continue

            # do task1 or task2
            for face in faces:
                # if task_2, it would be a face recognition task at first
                if task_2:
                    # do face recogntion
                    face_test_embedding = resnet(face).detach().cpu()
                    tmp = 0
                    for embedding in embeddings:
                        tmp += (face_test_embedding - embedding).norm().item()
                    score_avg = tmp / embeddings.shape[0]
                    # if score lower than yuzhi, it represents that the person is the one we wanna find
                    if score_avg < score_yuzhi:
                        # append to new_faces
                        new_faces.append(face)
                    else:
                        continue

                # if not task2, it should be task1
                else:
                    # append to new_faces
                    new_faces.append(face)

        # do deepfake detection
        # faces_tensor = torch.stack(new_faces)  # convert list to pytorch tensor
        inference_dataset = Deepfakes(data=new_faces, flag=True)
        inference_loader = DataLoader(dataset=inference_dataset,
                                      batch_size=32,
                                      pin_memory=True,
                                      num_workers=8,
                                      shuffle=False)
        pred_cls, fake_prob = deepfake_detection(inference_loader)

        # save the pred result of our model
        y_pred.append(pred_cls)
        y_true.append(label_value)
        print(f'{per_video_path} is done! pred: {y_pred[-1]}, true: {y_true[-1]}')

# cal auc metric for all videos
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print("finish!")
