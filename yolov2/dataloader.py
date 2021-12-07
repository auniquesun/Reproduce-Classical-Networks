from torchvision.transforms import transforms

from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection
"""
    CocoDetection Reference: https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html

    def __getitem__(self, index: int)

        return image, target

        @param: image  - an PIL image
        @param: target - annos of this image
"""


class CocoDataLoader(object):
    def __init__(self, args):
        self.tsfm = transforms.Compose(
                [
                    transforms.Resize((args.image_height, image_width)),
                    transforms.ToTensor(),
                ]
            )

    def get_trainloader(self, args):
        trainset = CocoDetection(root=args.images_root_path, annFile=args.label_file, transform=self.tsfm, target_transform=None, transforms=None)

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_gpus
        )

        return self.trainloader

    def get_testloader(self, args):
        testset = CocoDetection(root=args.images_root_path, annFile=args.label_file, transform=self.tsfm, target_transform=None, transforms=None)

        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_gpus
        )

        return self.testloader