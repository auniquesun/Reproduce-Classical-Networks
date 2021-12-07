import argparse



def parse_args():
    parser = argparse.ArgumentParser("Arguments of training YOLOv2 from scratch")

    parser.add_argument('--epochs', type=int, default=100, help='training epochs over the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu during training')
    parser.add_argument('--batch_size_test', type=int, default=32, help='batch size per gpu during test')
    parser.add_argument('--num_gpus')


    args = parser.parse_args()
    return args


def main():
    testloader = coco_dataloader.get_trainloader(args), coco_dataloader.get_testloader(args)