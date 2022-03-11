# distributed training on single node

import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
from torch import device, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value

from model import DenseNet121
from utils import AverageMeter, ProgressMeter, accuracy, adjust_lr, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Argument configurations for training DenseNet121 from scratch.')

    parser.add_argument('--epochs', type=int, default=90, help='training epochs over a whole dataset')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size per gpu during training')
    parser.add_argument('--num_workers', type=int, default=48, help='number of subprocesses to use for data loading')
    parser.add_argument('--lr', type=float, default=10e-1, help='initial learning rate of the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for learning rate of the optimizer')

    parser.add_argument('--seed', type=int, default=0, help='seed for initialize models with same weights on different GPUs')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed processes')

    parser.add_argument('--print_freq', type=int, default=500, help='evaluation frequency during training')
    parser.add_argument('--tensorboard', action='store_true', help='log progress using TensorBoard  ')

    parser.add_argument('--dataset_root', type=str, default='/mnt/sdb/public/data/common-datasets/imagenet', help='root path of the training dataset')
    parser.add_argument('--image_height', type=int, default=224, help='the height of image size input to DenseNet')
    parser.add_argument('--image_width', type=int, default=224, help='the width of image size input to DenseNet')

    parser.add_argument('--growth_rate', type=int, default=32, help='growth rate in Bottleneck layer of DenseNet')
    parser.add_argument('--compression_rate', type=float, default=0.5, help='compression rate in Transition layer of DenseNet')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of object categories in the training dataset')

    args = parser.parse_args()
    return args


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Training DenseNet on ImageNet should use GPU devices, but CUDA is unavailable!')

    if args.tensorboard:
        # model-BC-layers-growth_rate
        configure('runs/DenseNet-BC-121-32')   

    local_rank = args.local_rank
    ngpus = torch.cuda.device_count()

    main_worker(local_rank, ngpus, args)


def main_worker(local_rank, ngpus, args):
    best_prec1 = .0

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    
    print(f'local_rank: {local_rank}\n')

    torch.cuda.set_device(local_rank)

    # IMPORTANT: we need to set the random seed in each process so that the models are initialized with the same weights
    # Reference: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    # torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build model
    densenet = DenseNet121(in_channels=3, growth_rate=args.growth_rate, \
                           compression_rate=args.compression_rate, \
                           num_classes=args.num_classes)

    densenet.cuda(local_rank)

    densenet = nn.parallel.DistributedDataParallel(densenet, \
                                                    device_ids=[local_rank], \
                                                    output_device=local_rank)
    
    # Reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_width),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_root, 'train'), transform=train_transform)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_width),
        transforms.ToTensor(),
        normalize
    ])
    valset = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_root, 'val'), transform=val_transform)
    
    # num_replicas: int, Number of processes participating in distributed training. By default, world_size is retrieved from the current distributed group.
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=ngpus)
    batch_size = args.batch_size // ngpus
    num_workers = args.num_workers // ngpus

    train_data = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,  # when sampler is specified, shuffle should be False
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_data = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.SGD(densenet.parameters(), lr=args.lr, momentum=args.momentum, \
                            weight_decay=args.weight_decay)   

    # Reference: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
    # this is useful for cudnn finding optimal set of algorithms for particular configurations 
    # and accelerate training when the input sizes do not change over iteration.
    cudnn.backend = True

    for epoch in range(args.epochs):

        train_sampler.set_epoch(epoch)

        adjust_lr(args, optimizer, epoch)

        losses, top1, top5 = train(densenet, train_data, criterion, optimizer, epoch, local_rank, args)        

        if args.tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('top1_acc', top1.avg, epoch)
            log_value('top5_acc', top5.avg, epoch)

        # validate the model every epoch
        prec1 = validate(args, val_data, densenet, criterion, epoch)

        is_best = prec1.avg > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': densenet.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best)


def train(densenet, train_data, criterion, optimizer, epoch, local_rank, args):
    # enable train mode
    densenet.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # progress = ProgressMeter(len(train_data), 
    #                         [batch_time, data_time, losses, top1, top5],
    #                         prefix="Epoch: [{0}]".format(epoch))

    start = time.time()
    for i, (images, labels) in enumerate(train_data):
        # measure data loading time
        data_time.update(time.time() - start)

        # here images and labels are in BATCH-wise
        images, labels = images.cuda(local_rank), labels.cuda(local_rank)

        outputs = densenet(images)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        batch_time.update(end - start)

        if i % args.print_freq == 0:
            # i is the batch index
            # progress.display(i)
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_data), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses, top1, top5


def validate(args, val_data, densenet, criterion, epoch):
    """Reference: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    switch BatchNorm avg_pool max_pool layers in densenet to evaluate mode
    model.eval() is usually paired with torch.no_grad() to disable gradients computation
    """
    densenet.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # progress = ProgressMeter(len(val_data),
    #                         [batch_time, losses, top1, top5],
    #                         prefix="Test: [{0}]".format(epoch))

    start = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_data):
            images, labels = images.cuda(), labels.cuda()

            outputs = densenet(images)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            end = time.time()
            batch_time.update(end-start)

            if i % args.print_freq == 0:
                # progress.display(i)
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_data), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    msg_time = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    print(f'[{msg_time}]\n')

    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('top1_acc', top1.avg, epoch)
        log_value('top5_acc', top5.avg, epoch)

    return top1


if __name__ == '__main__':
    args = parse_args()

    main(args)
