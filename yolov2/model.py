import torch as t
import torch.nn as nn
import torch.nn.functional as F


class DarkNet19(nn.Module):
    def __init__():
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)
        # maxpool: stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        # maxpool: stride 2

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(128)
        # maxpool: stride 2

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv8_bn = nn.BatchNorm2d(256)
        # maxpool: stride 2

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv10_bn = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv12_bn = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv13_bn = nn.BatchNorm2d(512)
        # maxpool: stride 2

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv14_bn = nn.BatchNorm2d(1024)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv15_bn = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv16_bn = nn.BatchNorm2d(1024)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv17_bn = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv18_bn = nn.BatchNorm2d(1024)

        # above is first 18 layers of darknet, below is YOLOv2 head
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)
        self.conv19_bn = nn.BatchNorm2d(1024)
        self.conv20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)
        self.conv20_bn = nn.BatchNorm2d(1024)
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)
        self.conv21_bn = nn.BatchNorm2d(1024)
        # final 1x1 conv
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=425, kernel_size=1)
        self.conv22_bn = nn.BatchNorm2d(425)

    def forward(x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = self.conv8(x)
        x = self.conv8_bn(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = self.conv10(x)
        x = self.conv10_bn(x)
        x = self.conv11(x)
        x = self.conv11_bn(x)
        x = self.conv12(x)
        x = self.conv12_bn(x)
        x = self.conv13(x)
        x = self.conv13_bn(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv14(x)
        x = self.conv14_bn(x)
        x = self.conv15(x)
        x = self.conv15_bn(x)
        x = self.conv16(x)
        x = self.conv16_bn(x)
        x = self.conv17(x)
        x = self.conv17_bn(x)
        x = self.conv18(x)
        x = self.conv18_bn(x)

        x = self.conv19(x)
        x = self.conv19_bn(x)
        x = self.conv20(x)
        x = self.conv20_bn(x)
        x = self.conv21(x)
        x = self.conv21_bn(x)
        x = self.conv22(x)
        x = self.conv22_bn(x)
