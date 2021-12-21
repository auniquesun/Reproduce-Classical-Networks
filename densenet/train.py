import os
import argparse
import datetime

import torch
import torch.nn as nn
from torch import optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import DenseNet121

def parse_args():
    parser = argparse.ArgumentParser(help='Argument configurations for training DenseNet121 from scratch.')

    parser.add_argument('--epochs', type=int, default=90, help='training epochs over a whole dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu during training')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs used during training')
    parser.add_argument('--num_workers', type=int, default=12, help='number of subprocesses to use for data loading')
    parser.add_argument('--lr', type=float, default=10e-3, help='initial learning rate of the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of the optimizer')
    parser.add_argument('--multiprocess-distributed', action='store_true', help='use multiprocess to'  
                        'launch N processes per node, which has N GPUs')
    parser.add_argument('--eval_freq', type=int, default=1000, help='evaluation frequency during training')

    parser.add_argument('--data_path', type=str, default='/mnt/sdb/public/data/imagenet', help='root path of the training dataset')
    parser.add_argument('--image_height', type=int, default=448, help='the height of image size input to YOLOv2')
    parser.add_argument('--image_width', type=int, default=448, help='the width of image size input to YOLOv2')
    parser.add_argument('--image_root_path', type=str, default='/mnt/sdb/public/data/coco2017/train2017', help='COCO training images root path')
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
    ngpus_per_node = torch.cuda.device_count()

    # build model
    densenet = DenseNet121(in_channels=3, growth_rate=args.growth_rate, compression_rate=args.compression_rate, num_classes=args.num_classes)
    densenet.cuda()

    densenet = nn.parallel.DistributedDataParallel(densenet, devices=list(range(ngpus_per_node)))
    

    imagenet_data = torchvision.dataset.ImageNet(os.path.join(args.data_path), 'ILSVRC2012')

    # Reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_data = torch.utils.data.DataLoader(
        data=imagenet_data,
        split='train',
        batch_size=args.batch_size
        shuffle=True,
        num_workers=args.num_workers,
        transform=train_transform
    )

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    val_data = torch.utils.data.DataLoader(
        data=imagenet_data,
        split='val',
        batch_size=args.batch_size
        shuffle=False,
        num_workers=args.num_workers,
        transform=val_transform
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(densenet.parameters(), lr=args.lr, momentum=args.momentum, \
                            weight_decay=args.weight_decay)

    for i in range(args.epochs):
        for j, (images, labels) in enumerate(train_data):
            # here images and labels are in BATCH-wise
            images, labels = images.cuda(), labels.cuda()

            outputs = densenet(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validate every eval_freq
            if i % args.eval_freq == args.eval_freq-1:
                validate(val_data, densenet, criterion)


def validate(val_data, densenet, criterion):
    """Reference: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    switch BatchNorm avg_pool max_pool layers in densenet to evaluate mode
    model.eval() is usually paired with torch.no_grad() to disable gradients computation
    """
    densenet.eval()

    val_loss = .0
    incorrect = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_data):
            images, labels = images.cuda(), labels.cuda()

            outputs = densenet(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            pred = outputs.data.max(1)[1]

            # compute pred Not Equal label element-wise
            incorrect += pred.ne(labels.data).cpu().sum()

        val_loss /= len(val_data)
        total = len(val_data.dataset)

        msg_time = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        print(f'[{msg_time}]\n'
              f'validation loss: {val_loss}\nincorrect samples: '
              f'{incorrect}\ntotal samples: {total}\nerror rate: {incorrect/total}\n\n')


if __name__ == '__main__':
    args = parse_args()

    train(args)
