# -*-coding:utf-8-*-
import os
from pyexpat import model
import tempfile
import time
import argparse
from datetime import datetime
from matplotlib.pyplot import step
from numpy import save

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from early_stop import EarlyStopping

from dataset import Deepfakes
# from video_dataset import Deepfakes
from utils import *  # including other libraries like torch, tqdm, etc.
from multi_train_utils.distributed_utils import dist, cleanup, init_distributed_mode
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
from logger import prepare_logger

import gc
gc.enable()

label_dic = {'fake': 0, 'real': 1}
if torch.cuda.is_available() is False:
   raise EnvironmentError("not find GPU device for training.")
# # for single GPU
# cuda_num = os.environ['CUDA_VISIBLE_DEVICES']
# cuda_num_list = list(cuda_num.split(","))
# if len(cuda_num_list) == 1:
#    import torch.distributed as dist
# dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
# print("already init\n")

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
   elif model_name == 'mae':
      from vit_pytorch import ViT, MAE
      v = ViT(
         image_size = 256,
         patch_size = 32,
         num_classes = 1000,
         dim = 1024,
         depth = 6,
         heads = 8,
         mlp_dim = 2048
      )

      net = MAE(
         encoder = v,
         masking_ratio = 0.75,   # the paper recommended 75% masked patches
         decoder_dim = 512,      # paper showed good results with just 512
         decoder_depth = 6       # anywhere from 1 to 8
      )
   elif model_name == 'cipucnet':
      from model.CIPUCNet import CIPUCNet
      net = CIPUCNet(num_class=num_classes)
   elif model_name == 'efficient_vit':
      from model.efficientnet_vit import EfficientViT
      net = EfficientViT(image_size=224, num_classes=2)
   else:
      raise ValueError("model name error! plz try again.")
   return net


def main(args):
   global best_model_path
   setup_seed(20)
   assert os.path.exists(args.data_path), "dataset_path is not exists, plz try again."

   # dist mode
   init_distributed_mode(args=args)
   # rank = args.rank
   rank = 0
   device = torch.device(args.device)

   # update the learning rate based on gpu nums
   args.lr *= args.world_size

   # obtain dataset
   name = args.dataset_name

   # create a folder to save model
   save_folder = os.path.join(args.save_foler, args.model_name, name)
   os.makedirs(save_folder, exist_ok=True)

   # tensorboard
   if rank == 0:
      print(f'Start Tensorboard with "tensorboard --logdir={save_folder}", view at http://localhost:6006/')
      tb_writer = SummaryWriter(log_dir=save_folder, comment=f"{args.model_name}")

   # create logger
   log = prepare_logger(log_dir=save_folder, log_filename="log.txt", stdout=False)

   if rank == 0:
      log.info(args)
      log.info(f"Now, start training {name} dataset")

   train_dataset = Deepfakes(dataset_root=args.data_path, dataset_name=name, size=args.img_size,
                           frame_num=args.frame_num, mode='train', dataAug=args.aug)
   val_dataset = Deepfakes(dataset_root=args.data_path, dataset_name=name, size=args.img_size,
                           frame_num=args.frame_num, mode='val', dataAug=False)

   if rank == 0:
      log.info(f"Train Data: {len(train_dataset)}")
      log.info(f"Validation Data: {len(val_dataset)}")

   # 给每个rank对应的进程分配训练的样本索引
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
   val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

   # 将样本索引每batch_size个元素组成一个list
   train_batch_sampler = torch.utils.data.BatchSampler(
      train_sampler, args.batch_size, drop_last=True)

   if rank == 0:
      log.info('Using {} dataloader workers every process'.format(args.num_workers))

   # dataloader
   train_loader = DataLoader(dataset=train_dataset,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              # when setting batch_sampler, dont need batch_size, shuffle, sampler, and drop_last
                              batch_sampler=train_batch_sampler,) 
   val_loader = DataLoader(dataset=val_dataset,
                           batch_size=args.batch_size,
                           pin_memory=True,
                           num_workers=args.num_workers,
                           sampler=val_sampler,
                           shuffle=False,
                           drop_last=True,)

   # get model
   if rank == 0:
      log.info(f'Starting Loading {args.model_name} model')
   net = choose_model(model_name=args.model_name, num_classes=args.num_classes).to(device)

   checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
   # 不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
   if rank == 0:
      torch.save(net.state_dict(), checkpoint_path)
   dist.barrier()
   net.load_state_dict(torch.load(checkpoint_path, map_location=device))

   # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
   if args.syncBN:
      # 使用SyncBatchNorm后训练会更耗时
      net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)

   # 转为DDP模型
   net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)

   # AdamW
   optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay=args.weight_decay)

   # Scheduler
   if args.lr_decay == 'warmup': 
      scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
   elif args.lr_decay == 'multi':
      scheduler = MultiStepLR(optimizer, milestones=[8, 14, 20, 26], gamma=0.5)  # shanzhuo method
   elif args.lr_decay == 'step':
      scheduler = StepLR(optimizer, step_size=15, gamma=0.1)  # shanzhuo method
   else:
      raise NotImplementedError("Other methods are not implemented yet")

   # loss_function
   loss_function = torch.nn.CrossEntropyLoss()

   # initialize the early_stopping object
   early_stopping = EarlyStopping(patience=5)

   # train and val
   best_acc = 0.0
   for epoch in range(args.epoch):
      train_sampler.set_epoch(epoch)

      mean_loss = train_one_epoch(model=net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

      scheduler.step()

      sum_num = evaluate(model=net,
                           data_loader=val_loader,
                           device=device)
      acc = sum_num / val_sampler.total_size

      if rank == 0:
         log.info(f"[epoch {epoch+1}] accuracy: {round(acc, 3)}, mean_loss: {round(mean_loss, 3)}, lr: {round(optimizer.param_groups[0]['lr'], 7)}")
         # tensorboard vis
         tags = ["loss", "accuracy", "learning_rate"]
         tb_writer.add_scalar(tags[0], mean_loss, epoch)
         tb_writer.add_scalar(tags[1], acc, epoch)
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
         if acc > best_acc:
               best_acc = acc
               checkpoint = {'model': net,
                           'model_state_dict': net.state_dict(),
                           # 'optimizer_state_dict': optimizer.state_dict(),
                           'epoch': epoch}
               best_model_path = os.path.join(save_folder, 'best.pth')
               log.info(f"temporal best epoch: {epoch + 1}")
               torch.save(checkpoint, best_model_path)
         
         # early stop
         early_stopping(mean_loss, net)
        
         if early_stopping.early_stop:
            log.info("Early stopping")
            break
      
   log.info(f"Finished Training, best_acc = {round(best_acc, 3)}, "
         f"dataset_name= {name}, model_name={args.model_name}")

   # clear cache
   if rank == 0:
      if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
   cleanup()

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
   parser.add_argument('--batch_size', type=int, default=100)
   parser.add_argument('--lr', type=float, default=0.0001)
   parser.add_argument('--aug', type=bool, default=True)
   parser.add_argument('--num_classes', type=int, default=2)
   parser.add_argument('--num_workers', type=int, default=4)
   parser.add_argument('--save_foler', type=str, default='weights')
   parser.add_argument('--model_name', type=str, default='vit')
   parser.add_argument('--dataset_name', type=str, default='Deepfakes')
   parser.add_argument('--lr_decay', type=str, default='warmup')  # warmup or multi or step
   parser.add_argument('--weight_decay', type=float, default=1e-2)
   parser.add_argument('--frame_num', type=int, default=100)
   parser.add_argument('--img_size', type=int, default=224)
   parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/FF_data/FF_HQ/train_face')
   # gpu-params
   parser.add_argument('--syncBN', type=bool, default=False)
   parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
   parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
   parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
   opt = parser.parse_args()

   # run
   main(opt)
   # os.system("shutdown")
