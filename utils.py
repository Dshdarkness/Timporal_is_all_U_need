# -*-coding:utf-8-*-
import os
import json
import pickle
import random
import torch
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

device = "cuda"

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

def plot_confusion_matrix(y_true, y_pred, d, save_folder, title='Confusion Matrix', cmap=plt.cm.ocean):
    labels = d.keys()
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)  
    np.set_printoptions(precision=2)

    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')

    title = 'Normalized confusion matrix'
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=45)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'), format='png')
    #plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function, logger):
    model.train()
    loss_function = loss_function
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device) 
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        # grad cliping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100, norm_type=2)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            logger.info('WARNING: non-finite loss, ending training ', loss)
            return None, None

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_function, ema, logger):
    loss_function = loss_function

    model.eval()

    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device) 

    sample_num = 0
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        if ema is not None and epoch > 4:
            # Validation: with EMA
            # the .average_parameters() context manager
            # (1) saves original parameters before replacing with EMA version
            # (2) copies EMA parameters to model
            # (3) after exiting the `with`, restore original parameters to resume training later
            with ema.average_parameters():
                pred = model(images.to(device))
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()

                loss = loss_function(pred, labels.long().to(device))
        
        # evaluate without ema
        else:
            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.long().to(device))
        
        if not torch.isfinite(loss):
            logger.info('WARNING: non-finite loss, ending validation ', loss)
            return None, None

        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
