#!/usr/bin/python
# -*-coding: utf-8 -*-
"""Use for the image classification with different Resnet on the calth256 dataset (257 classes)
    Get the Training time cost and the inference time and the parameters on the total same GPU devices.
    Training use the batch iter for validation
"""
from __future__ import print_function

import os
import math
import time
import datetime

import json
import shutil
import random
import argparse
import logging

import torch
import numpy as np
import horovod.torch as hvd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from PIL import ImageFile
from tensorboardX import SummaryWriter
from models.build_model import BuildModel
from torchvision import datasets, transforms, models
from dataset.imagenet_dataset import ImageNetTrainingDataset, ImageNetValidationDataset

import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Training settings
parser = argparse.ArgumentParser(description='cub dataset for FGVC training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrainmodel', default=os.path.expanduser('/data/remote/output_ckpt_with_log_for_model_with_trick/ckpt/resnet_in_dataset_0430_ckpt/checkpoint-71.pth.tar'),
                    help='The pretrain model for model init the weights')
parser.add_argument('--train-dir', default=os.path.expanduser('/data/remote/code/sex_image_classification/data/sex_data/sample_train.txt'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/data/remote/code/sex_image_classification/data/sex_data/validation.txt'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='/data/remote/code/sex_image_classification/output_dir/output_log/2020_4_13_sample_train_logdir',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='/data/remote/code/sex_image_classification/output_dir/output_checkpoint/2020_4_13_sample_train_checkpoint',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--fp16', type=int, default=0,
                    help="use the fp16, model half and bn float32")
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=256,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--num_thread', type=int, default=4,
                    help="numworkers thread for process")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# System need path
parser.add_argument('--data', type=str, default='',
                    help='Choice a data for training plartform')
parser.add_argument('--out-dir', type=str, default="/data/remote/code/small_classifcation_code/output_dir",
                    help="System need")
parser.add_argument('--hdfs-namenod', type=str, default="hdfs://10.26.133.3:8020/",
                    help="System need for hdfs data distributed")

# model parser
parser.add_argument('--classes_weights', type=int, default=1,
                    help="Use the weight cross entropy")
parser.add_argument('--num_classes', type=int, default=5,
                    help="num classes for image classification")
parser.add_argument('--model_name', type=str, default="resnet50",
                    help="choice the model name")
parser.add_argument('--pretrain', type=bool,
                    help="Pretrain the weight")
parser.add_argument('--imagenet_pretrain', type=int, default=1,
                    help="Use the model original Imagenet pretrain")
parser.add_argument('--sample_type', type=str, default="",
                    help="Sample type for dataset sampler in ['balance', 'reverse', 'weights', 'uniform', 'origin']")
parser.add_argument('--model_freeze', type=int, default=0,
                    help="Freeze the model with feature exterctor")
parser.add_argument('--optimizer', type=str, default='sgd',
                    help="Choice the optimizer for training!")
parser.add_argument('--finetune', type=int, default=0,
                    help="Choice the optimizer for training!")
# rank 0: average metrics with rank 0 for distributed training
parser.add_argument('--cosine_lr', type=int, default=0,
                    help="cosine lr for training loop")


class Metric_rank:
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1
        # print(self.sum)
        # print(self.n)

    @property
    def average(self):
        return self.sum / self.n


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_with_iter(epoch, interval, batch_iter):
    """
        Training with the iter for validation
        Args:
            epoch (int): The epoch
            interval (int): the interval for validation with batch
            batch_iter (int): get the batch iter
    """
    alpha = 1
    model.train()
    # set the random seed for torch.generator() to shuffle the dataset
    trainSampler.set_epoch(epoch)
    if hvd.rank() == 0:
        print("="*50)

    epoch_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()

        if args.finetune:
            adjust_learning_rate_for_finetune(epoch, batch_idx)
        elif args.cosine_lr:
            adjust_learning_rate_for_cosine_decay(epoch, batch_idx)
        else:
            adjust_learning_rate(epoch, batch_idx)

        if args.cuda:
            if not args.fp16:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.half().cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # training batch acc
        train_acc = accuracy(output, target)

        batch_iter += 1
        if hvd.rank() == 0:
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]

            waste_time = time.time() - batch_start
            print("Training Epoch: [{}/{}] batch: [{}/{}] batchiter: [{}/{}] Loss: {:.4f} Accuracy_lv1: {:.4f} Learning_rate: {:.6f} Time: {:.2f} date: {}".format(
                epoch, args.epochs, batch_idx+1, total_train_sampler, batch_iter, total_train_sampler *
                args.epochs, loss.item(), train_acc.item(
                ), learning_rate, waste_time, str(datetime.datetime.now())
            ))

            # train log writer
            if log_writer:
                # train batch
                log_writer.add_scalars(
                    'train/lv1', {
                        'loss': loss.item(),
                        'acc': train_acc.item()
                    }, batch_iter
                )

                log_writer.add_scalar(
                    'train/batch_time', waste_time, batch_iter
                )
                log_writer.add_scalar(
                    'learning_rate', learning_rate, batch_iter)

    # validaiton with each epoch
    if args.val_dir is not None or args.val_dir != "":
        validation_rank, val_acc = validatin_acc()
        if hvd.rank() == 0:
            print("Validation Epoch: [{}/{}] batchiter: [{}/{}] Loss: {:.4f} RankLoss: {:.4f} Accuracy: {:.4f} Time: {:.2f}".format(
                epoch, args.epochs, batch_iter, total_train_sampler *
                args.epochs, validation_rank["loss"], validation_rank["rank_loss"], val_acc["val_acc"], time.time(
                ) - batch_start
            ))

            # validation_log
            if log_writer:
                log_writer.add_scalars(
                    'Val/batch', {
                        'rank_loss': validation_rank["rank_loss"],
                        'loss': validation_rank["loss"],
                    },
                    batch_iter
                )
                log_writer.add_scalars(
                    'Val/batch_acc', {
                        'accuracy': val_acc['val_acc']
                    },
                    batch_iter
                )

                log_writer.add_scalars(
                    'Val/epoch_acc', {
                        'accuracy': val_acc['val_acc']
                    },
                    epoch + 1
                )

    # save checkpoint with the epoch
    save_checkpoint(epoch, "epoch")

    if hvd.rank() == 0:
        print("Epoch [{}/{}] waste time is {}".format(epoch,
                                                      args.epochs, time.time() - epoch_start))

    return batch_iter


def validatin_acc():
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    val_rank_loss = Metric_rank('val_rank_loss')
    # val_rank_accuarcy = Metric_rank('val_rank_accuarcy')

    total_correct_sum = torch.tensor(0.0)
    total_number = torch.tensor(0.0)

    with torch.no_grad():
        # start
        for batch_idx, loader in enumerate(val_loader):
            data, target = loader[0], loader[1]
            total_number += target.size()[0]
            if args.cuda:
                # fp16
                if not args.fp16:
                    data, target = data.cuda(), target.cuda()
                else:
                    data, target = data.half().cuda(), target.cuda()
            # calculate the lv1 output
            output = model(data)

            loss = F.cross_entropy(output, target)
            val_loss.update(loss)
            val_correct_sum = accuracy_for_validation(output, target)
            
            # hvd_total_correct_sum = hvd.allreduce(total_correct_sum) * hvd.size()
            # validation_accuracy = float(
            # hvd_total_correct_sum.item() / (total_number.item() * hvd.size()))
            # 计算正确的累加的个数
            total_correct_sum += val_correct_sum
            validation_loss_ = 0.0
            if hvd.rank() == 0:
                validation_loss_ = loss.item()

            val_rank_loss.update(validation_loss_)
        
        hvd_total_correct_sum = hvd.allreduce(total_correct_sum) * hvd.size()
        validation_accuracy = float(hvd_total_correct_sum.item() / (total_number.item() * hvd.size()))

        # validation_accuracy = float(
        #     total_correct_sum.item() / total_number.item()
        # )

        validation_rank_avg_loss = val_rank_loss.average
        validation_loss = val_loss.avg

        validation_rank = {
            'rank_loss': validation_rank_avg_loss,
            'loss': validation_loss
        }

        val_acc = {
            'val_acc': validation_accuracy
        }

    # train
    model.train()
    return validation_rank, val_acc


def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < int(args.epochs * 0.3):
        lr_adj = 1.
    elif epoch < int(args.epochs * 0.6):
        lr_adj = 1e-1
    elif epoch < int(args.epochs * 0.9):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * \
            hvd.size() * args.batches_per_allreduce * lr_adj


# learning rate adjust for cosine decay
def adjust_learning_rate_for_cosine_decay(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    else:
        batch_sample = total_train_sampler * \
            (epoch - args.warmup_epochs) + batch_idx
        lr_adj = 1/2 * (1 + math.cos(batch_sample *
                                     math.pi / total_sampler_batch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * \
            hvd.size() * args.batches_per_allreduce * lr_adj


def adjust_learning_rate_for_finetune(epoch, batch_idx):
    if epoch < int(args.epochs * 0.8):
        lr_adj = 1.
    else:
        lr_adj = 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * \
            hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()


def accuracy_for_validation(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().sum().item()


def save_checkpoint(number, batch_epoch):
    if hvd.rank() == 0:

        if batch_epoch == "epoch":
            filepath = os.path.join(
                checkfile_dir, "checkpoint-epoch-{}.pth.tar".format(number))
        else:
            filepath = os.path.join(
                checkfile_dir, "checkpoint-batch-{}.pth.tar".format(number))
        # filepath = args.checkpoint_format.format(epoch=epoch + 1)
        cpu_state_dict = {}
        for key, value in model.state_dict().items():
            cpu_state_dict[key] = value.cpu()

        state = {
            'model': cpu_state_dict,
            'optimizer': optimizer.state_dict(),
        }
        # if epoch % interepoch == 0:
        torch.save(state, filepath)

class Metric(object):

    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        if args.fp16:
            self.sum += hvd.allreduce(val.detach().cpu().float(), name=self.name)
            self.n += 1
        else:
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
            self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

if __name__ == "__main__":

    args = parser.parse_args()
    setup_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce
    best_validation_acc = 0.0
    checkfile_dir = args.checkpoint_format
    torch.autograd.set_detect_anomaly(True)
    # init the horovod
    hvd.init()
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        resume_checkpoint = os.path.join(
            args.checkpoint_format, "checkpoint-{epoch}.pth.tar".format(epoch=try_epoch))
        if os.path.exists(resume_checkpoint):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0, name='resume_from_epoch').item()
    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(args.num_thread)
    kwargs = {'num_workers': args.num_thread,
              'pin_memory': True} if args.cuda else {}
    # training dataset
    trainDataset = ImageNetTrainingDataset(args.train_dir)
    trainSampler = torch.utils.data.distributed.DistributedSampler(
        trainDataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=allreduce_batch_size,
        sampler=trainSampler,
        **kwargs
    )

    if args.val_dir is not None or args.val_dir != "":
        valDataset = ImageNetValidationDataset(args.val_dir)
        valSampler = torch.utils.data.distributed.DistributedSampler(
            valDataset,
            num_replicas=hvd.size(),
            rank=hvd.rank(),
            shuffle=False
        )

        val_loader = torch.utils.data.DataLoader(
            valDataset,
            batch_size=allreduce_batch_size,
            sampler=valSampler,
            **kwargs
        )

    # 训练数据的迭代次数
    total_train_sampler = 0
    train_iter = len(trainDataset) / (args.batch_size * hvd.size())
    if train_iter - int(train_iter) > 0.0:
        total_train_sampler = int(train_iter) + 1
    else:
        total_train_sampler = int(train_iter)

    # 验证数据的迭代次数
    if args.val_dir is not None or args.val_dir != "":
        total_validation_sampler = 0
        val_iter = len(valDataset) / (args.batch_size * hvd.size())
        if val_iter - int(val_iter) > 0.0:
            total_validation_sampler = int(val_iter) + 1
        else:
            total_validation_sampler = int(val_iter)

    # imagnet pretrain model
    if args.imagenet_pretrain:
        imagenet_pretrain = True
        if hvd.rank() == 0:
            print("Use the imagenet pretrain")
    else:
        imagenet_pretrain = False
        if hvd.rank() == 0:
            print("Not use the imagenet pretrain")
    build_model = BuildModel(args.model_name, args.num_classes, imagenet_pretrain)
    model = build_model()
    if hvd.rank() == 0:
        print(model)
    # load the pretrain model weights
    # if hvd.rank() == 0:
    if args.pretrainmodel.endswith("tar"):
        state_dict = torch.load(args.pretrainmodel, map_location="cpu")
        model.load_state_dict(state_dict["model"])
    elif args.pretrainmodel.endswith("pkl"):
        state_dict = torch.load(args.pretrainmodel, map_location="cpu")
        model.load_state_dict(state_dict)
    if hvd.rank() == 0:
        print("Load the pretrian model {}".format(args.pretrainmodel))

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * \
        hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    if args.model_freeze:
        # if args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=(args.base_lr * lr_scaler),
                              momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    else:
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=(args.base_lr *
                                      lr_scaler),
                                  momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=(args.base_lr*lr_scaler),
                weight_decay=args.wd
            )

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # use the default parpameters with hvd distributed optmizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
    # print("optimizer:", optimizer)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    # resume_from_epoch = 0

    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = os.path.join(
            args.checkpoint_format, "checkpoint-{epoch}.pth.tar".format(epoch=resume_from_epoch))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    if hvd.size() > 1:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if hvd.rank() == 0:
        print("-"*50)
        print("Training Information!!!")
        information_kwargs = {
            "Epoch ": args.epochs,
            "Model ": args.model_name,
            "BatchSize ": args.batch_size,
            "Classes ": args.num_classes,
            "Optimizer_learning_rate ": args.base_lr * lr_scaler,
            "HVD size ": hvd.size(),
            "PretrainModel": args.pretrainmodel,
            "Log dir ": args.log_dir,
            "FP16 ": args.fp16,
            "Save Checkpoint ": args.checkpoint_format,
            "Train_data ": args.train_dir,
            "Validation_data ": args.val_dir,
            "num_wrokers ": args.num_thread,
            "Use ImageNet Pretrain ": imagenet_pretrain,
            "Optimizer ": args.optimizer,
            "loss_function ": "softmax"
        }
        Information_data = json.dumps(
            information_kwargs, sort_keys=True, indent=4, separators=(',', ':'))
        print(Information_data)

    # batch 迭代次数
    batch_iter = 0
    T = int(args.epochs - args.warmup_epochs)
    total_sampler_batch = T * total_train_sampler
    # training
    for epoch in range(resume_from_epoch, args.epochs):
        batch_iter = train_with_iter(epoch, 0, batch_iter)
    
    print("Fnish the train loop!!!")
