# -*-coding:utf-8-*-
import os
import time
import argparse
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from early_stop import EarlyStopping

from utils import *  # including other libraries like torch, tqdm, etc.
from logger import prepare_logger
from torch_ema import ExponentialMovingAverage

import gc
gc.enable()

label_dic = {'fake': 0, 'real': 1}
if torch.cuda.is_available() is False:
   raise EnvironmentError("not find GPU device for training.")

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True


def choose_model(model_name: str, num_classes: int):
   global net
   if model_name == 'xception':
      from model.xceptionCNN import Xception
      net = Xception(num_classes=num_classes)
      state_dict = torch.load('pretrained/xception-b5690688.pth')
      for name, weights in state_dict.items():
         if 'pointwise' in name:
               state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
      net.load_state_dict(state_dict, False)
   elif model_name == 'mesonet':
      from model.mesonet import Meso4
      net = Meso4(num_classes=num_classes)
   elif model_name == 'mesonet_incep':
      from model.mesonet import MesoInception4
      net = MesoInception4(num_classes=num_classes)
   elif model_name == 'f3net':
      from model.f3net import F3Net
      net = F3Net(num_classes=num_classes, img_width=320, img_height=320)
   elif model_name == 'cnn_rnn':
      from model.resnext_lstm import ResNext_LSTM
      net = ResNext_LSTM(num_classes, latent_dim=2048, lstm_layers=3,
                        hidden_dim=256, bidirectional=False)
   elif model_name == 'cnn_lstm':
      from model.cnn_lstm import cnn_lstm
      net = cnn_lstm(num_class=2)
   elif model_name == 'vit':
      from model.vit_model import vit_base_patch16_224_in21k as create_model
      net = create_model(num_classes=2, has_logits=False)
   elif model_name == 'cipucnet':
      from model.CIPUCNet_gnn import CIPUCNet
      net = CIPUCNet(num_class=num_classes)
   elif model_name == 'efficient_vit':
      from model.efficientnet_vit import EfficientViT
      net = EfficientViT(image_size=224, num_classes=2)
   elif model_name == 'lrcn':
      from model.LRCN import LRCN
      net = LRCN(512, 2)
   elif model_name == 'mgat':
      from model.MGAT import MGAT
      net = MGAT(num_class=num_classes)
   elif model_name == 'hff':
      from model.hff_net.hff import Two_Stream_Net
      net = Two_Stream_Net()  #2021cvpr
   elif model_name == 'multiple_attention':
      from model.multiple_attention.MAT import MAT
      net = MAT() #2021cvpr
   else:
      raise ValueError("model name error! plz try again.")
   return net


def main(args):
   global best_model_path
   setup_seed(20)
   assert os.path.exists(args.data_path), "dataset_path is not exists, plz try again."

   device = torch.device(args.device)

   # obtain dataset
   name = args.dataset_name

   # create a folder to save model
   save_folder = os.path.join(args.save_foler, args.model_name, name)
   os.makedirs(save_folder, exist_ok=True)

   # tensorboard
   print(f'Start Tensorboard with "tensorboard --logdir={save_folder}", view at http://localhost:6006/')
   tb_writer = SummaryWriter(log_dir=save_folder, comment=f"{args.model_name}")

   # create logger
   log = prepare_logger(log_dir=save_folder, log_filename="log.txt", stdout=False)

   log.info(args)
   log.info(f"Now, start training {name} dataset")

   # get dataset according to the type of model (cnn or rnn type)
   if args.temporal:
      ## temporal-based model
      from video_dataset import Deepfakes
   ## spatial-based model or frequency-based model
   else:
      from dataset import Deepfakes
   train_dataset = Deepfakes(dataset_root=args.data_path, dataset_name=name, size=args.img_size,
                           frame_num=args.frame_num, mode='train', dataAug=args.aug)
   val_dataset = Deepfakes(dataset_root=args.data_path, dataset_name=name, size=args.img_size,
                           frame_num=args.frame_num, mode='val', dataAug=False)

   if args.temporal:
      log.info(f"Train Data: {len(train_dataset) * args.frame_num}")
      log.info(f"Validation Data: {len(val_dataset) * args.frame_num}")
   else:
      log.info(f"Train Data: {len(train_dataset)}")
      log.info(f"Validation Data: {len(val_dataset)}")

   log.info('Using {} dataloader workers every process'.format(args.num_workers))

   # dataloader
   train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  drop_last=True,)
   val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                shuffle=False,
                                drop_last=True,)

   # get model
   log.info(f'Starting Loading {args.model_name} model')
   net = choose_model(model_name=args.model_name, num_classes=args.num_classes)

   if torch.cuda.is_available():
        # transfer data from cpu to device
        net.to(device)

   optimizer =torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
   # AdamW
   # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, betas=(0.9, 0.999), \
   #                               eps=1e-08, amsgrad=False, weight_decay=args.weight_decay)

   # Scheduler
   if args.lr_decay == 'warmup': 
      scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)
   elif args.lr_decay == 'multi':
      scheduler = MultiStepLR(optimizer, milestones=[8, 14, 20, 26], gamma=0.5)  # shanzhuo method
   elif args.lr_decay == 'step':
      scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
   else:
      raise NotImplementedError("Other methods are not implemented yet")
   
   if args.use_ema:
      log.info("let's use ema for params update")
      ema = ExponentialMovingAverage(net.parameters(), decay=0.995)

   # loss_function
   loss_function = torch.nn.CrossEntropyLoss()
   # loss_function = LabelSmoothingCrossEntropy()

   # initialize the early_stopping object
   early_stopping = EarlyStopping(patience=10)

   # train and val
   best_acc = 0.0
   for epoch in range(args.epoch):
      train_loss, train_acc = train_one_epoch(model=net,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=args.device,
                                                epoch=epoch,
                                                loss_function=loss_function,
                                                logger=log,)
      log.info(f"[epoch {epoch+1}] train_accuracy: {round(train_acc, 3)}, train_mean_loss: {round(train_loss, 3)}, lr: {round(optimizer.param_groups[0]['lr'], 7)}")

      scheduler.step()
      if args.use_ema and epoch > 4:
         # Update the moving average with the new parameters from the last optimizer step
         log.info(f"Use ema for params update, epoch: {epoch+1}")
         ema.update()

      val_loss, val_acc = evaluate(model=net,
                                    data_loader=val_loader,
                                    device=args.device,
                                    epoch=epoch, loss_function=loss_function,
                                    ema=None if not args.use_ema else ema,
                                    logger=log,)

      log.info(f"[epoch {epoch+1}] val_accuracy: {round(val_acc, 3)}, val_mean_loss: {round(val_loss, 3)}")
      # tensorboard vis
      tags = ["loss", "accuracy", "learning_rate"]
      tb_writer.add_scalar(tags[0], train_loss, epoch)
      tb_writer.add_scalar(tags[1], val_acc, epoch)
      tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

      # save the last model
      checkpoint = {'model': net,
                'model_state_dict': net.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}
      last_model_path = os.path.join(save_folder, 'last.pth')
      # log.info(f"last epoch save: {epoch + 1}")
      torch.save(checkpoint, last_model_path)

      # save the best model
      if val_acc > best_acc:
        best_acc = val_acc
        checkpoint = {'model': net,
                    'model_state_dict': net.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}
        best_model_path = os.path.join(save_folder, 'best.pth')
        log.info(f"temporal best epoch: {epoch + 1}")
        torch.save(checkpoint, best_model_path)
    
      # early stop
      early_stopping(train_loss, net)

      if early_stopping.early_stop:
          log.info("Early stopping")
          break
      
   log.info(f"Finished Training, best_acc = {round(best_acc, 3)}, "
           f"dataset_name= {name}, model_name={args.model_name}")

   # clear memory
   del net, train_loader, val_loader, \
      optimizer, scheduler, loss_function
   gc.collect()
   torch.cuda.empty_cache()
   time.sleep(5)


if __name__ == '__main__':
   # params settting
   parser = argparse.ArgumentParser()
   # hyper-params
   parser.add_argument('--epoch', type=int, default=30)
   parser.add_argument('--batch_size', type=int, default=48)
   parser.add_argument('--lr', type=float, default=0.00008)
   parser.add_argument('--aug', type=bool, default=True)
   parser.add_argument('--num_classes', type=int, default=2)
   parser.add_argument('--num_workers', type=int, default=4)
   parser.add_argument('--save_foler', type=str, default='weights/LQ_2AE')
   parser.add_argument('--model_name', type=str, default='multiple_attention')
   parser.add_argument('--dataset_name', type=str, default='Deepfakes')
   parser.add_argument('--use_ema', type=bool, default=False)
   parser.add_argument('--lr_decay', type=str, default='step')  # warmup or multi or step
   parser.add_argument('--temporal', type=bool, default=False)
   parser.add_argument('--weight_decay', type=float, default=3e-3)
   parser.add_argument('--frame_num', type=int, default=1)
   parser.add_argument('--img_size', type=int, default=320)
   parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/FF_data/FF_LQ/train_face')
   # gpu-params
   parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
   opt = parser.parse_args()

   # run
   main(opt)
   # os.system("shutdown")
