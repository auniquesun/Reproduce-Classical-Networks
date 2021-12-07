import argparse

from dataloader import CocoDataLoader
from optimizer import Optimizer
from loss_util import Loss
from model import DarkNet19
from logs import Logger


def parse_args():
    parser = argparse.ArgumentParser("Arguments of training YOLOv2 from scratch")

    parser.add_argument('--epochs', type=int, default=100, help='training epochs over the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu during training')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs used during training')
    parser.add_argument('--lr', type=float, default=10e-3, help='initial learning rate of the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of the optimizer')
    parser.add_argument('--lambda_cls', type=float, default=0.9, help='weight of object classification loss')
    parser.add_argument('--lambda_box', type=float, default=0.9, help='weight of object bbox loss')
    parser.add_argument('--lambda_objectness', type=float, default=0.9, help='weight of objectness loss')

    parser.add_argument('--image_height', type=int, default=448, help='the height of image size input to YOLOv2')
    parser.add_argument('--image_width', type=int, default=448, help='the width of image size input to YOLOv2')
    parser.add_argument('--image_root_path', type=str, default='/mnt/sdb/public/data/coco2017/train2017', help='COCO training images root path')
    parser.add_argument('--label_file', type=str, default='/mnt/sdb/public/data/coco2017/annotations/instances_train2017.json', \
                        help='the label file of training set')

    parser.add_argument('--log_file', type=str, default='output/log.txt', help='path to the log file')

    args = parser.parse_args()
    return args


def main(args):
    coco_dataloader = CocoDataLoader(args)
    trainloader = coco_dataloader.get_trainloader(args)

    yolov2 = DarkNet19()
    optimizer = Optimizer(yolov2.parameters(), args)
    criterion = Loss()

    logger = Logger()
    logger.info('----- Starting training -----')

    for epoch in range(args.epochs):

        for i,data in enumerate(trainloader):
            images, targets = data

            optimizer.zero_grad()

            outputs = yolov2(images)
            loss = criterion(outputs, targets, args)

            loss.backward()
            optimizer.step()

            logger.info(f'Epoch: {epoch+1}/{args.epochs}, Step: {i+1}, Loss: {loss.data}')

    logger.info('----- Training done! -----')


if '__main__' == __name__:
    args = parse_args()
    
    main(args)