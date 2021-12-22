import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
from torch import optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import DenseNet121

from utils import AverageMeter, accuracy, adjust_lr, save_checkpoint

from tensorboard_logger import configure, log_value


def parse_args():
    parser = argparse.ArgumentParser(description='Argument configurations for training DenseNet121 from scratch.')

    parser.add_argument('--epochs', type=int, default=90, help='training epochs over a whole dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu during training')
    parser.add_argument('--num_workers', type=int, default=12, help='number of subprocesses to use for data loading')
    parser.add_argument('--lr', type=float, default=10e-3, help='initial learning rate of the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for learning rate of the optimizer')

    parser.add_argument('--world-size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--multiprocess-distributed', action='store_true', help='use multiprocess to'  
                        'launch N processes per node, which has N GPUs')
    parser.add_argument('--print_freq', type=int, default=1000, help='evaluation frequency during training')
    parser.add_argument('--tensorboard', action='store_true', help='log progress using TensorBoard  ')

    parser.add_argument('--dataset_root', type=str, default='/mnt/sdb/public/data/imagenet', help='root path of the training dataset')
    parser.add_argument('--image_height', type=int, default=448, help='the height of image size input to YOLOv2')
    parser.add_argument('--image_width', type=int, default=448, help='the width of image size input to YOLOv2')
    parser.add_argument('--label_file', type=str, default='/mnt/sdb/public/data/coco2017/annotations/instances_train2017.json', \
                        help='the label file of training set')

    parser.add_argument('--growth_rate', type=int, default=32, help='growth rate in Bottleneck layer of DenseNet')
    parser.add_argument('--compression_rate', type=float, default=0.5, help='compression rate in Transition layer of DenseNet')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of object categories in the training dataset')

    parser.add_argument('--log_file', type=str, default='output/log.txt', help='path to the log file')

    args = parser.parse_args()
    return args


def train(args):
    
    if not torch.cuda.is_available():
        raise RuntimeError('Training DenseNet on ImageNet should use GPU devices, but CUDA is unavailable!')

    if args.tensorboard:
        configure('runs/DenseNet-BC-121-32')

    # build model
    densenet = DenseNet121(in_channels=3, growth_rate=args.growth_rate, compression_rate=args.compression_rate, num_classes=args.num_classes)
    densenet.cuda()

    ngpus_per_node = torch.cuda.device_count()
    densenet = nn.parallel.DistributedDataParallel(densenet, devices=list(range(ngpus_per_node)))
    
    # Reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    trainset = torchvision.dataset.ImageFolder(root=os.path.join(args.dataset_root, 'train'), transform=train_transform)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    valset = torchvision.dataset.ImageFolder(root=os.path.join(args.dataset_root, 'val'), transform=val_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    train_data = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size
        shuffle=False,  # when sampler is specified, shuffle should be False
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_data = torch.utils.data.DataLoader(
        data=valset,
        batch_size=args.batch_size
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(densenet.parameters(), lr=args.lr, momentum=args.momentum, \
                            weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    best_prec1 = .0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        adjust_lr(args, optimizer, epoch)

        start = time.time()
        for i, (images, labels) in enumerate(train_data):
            # measure data loading time
            data_time.update(time.time() - start)

            # here images and labels are in BATCH-wise
            images, labels = images.cuda(), labels.cuda()

            outputs = densenet(images)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data[0], images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time()
            batch_time.update(end - start)

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_data), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        if args.tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_acc', top1.avg, epoch)

        # validate the model every epoch
        prec1 = validate(args, val_data, densenet, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': densenet.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best)


def validate(args, val_data, densenet, criterion):
    """Reference: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    switch BatchNorm avg_pool max_pool layers in densenet to evaluate mode
    model.eval() is usually paired with torch.no_grad() to disable gradients computation
    """
    densenet.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_data):
            images, labels = images.cuda(), labels.cuda()

            outputs = densenet(images)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            end = time.time()
            batch_time.update(end-start)

            if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg
    # msg_time = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    # print(f'[{msg_time}]\n'
    #         f'validation loss: {val_loss}\nincorrect samples: '
    #         f'{incorrect}\ntotal samples: {total}\nerror rate: {incorrect/total}\n\n')


if __name__ == '__main__':
    args = parse_args()

    train(args)
