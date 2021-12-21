import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        """
        @param: in_channels - the number of channels of input feature maps to Bottleneck
        @param: growth_rate - the number of channels of output feature maps of Bottleneck
        """
        super(Bottleneck, self).__init__()

        # In DenseNet, each 1x1 conv produces 4k feature maps, here k is the growth_rate
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4*growth_rate, kernel_size=1, \
                                bias=False)
        self.bn1 = nn.BatchNorm2d(4*growth_rate)

        self.conv2 = nn.Conv2d(in_channels=4*growth_rate, out_channels=growth_rate, kernel_size=3, \
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate)

    def forward(x):
        """
        each Bottleneck Layer consists of two conv operations: 1x1 conv and 3x3 conv
        """
        out = self.conv1(F.ReLu(self.bn1(x)))
        out = self.conv2(F.ReLu(self.bn2(out)))
        # concatenate tensors in COLUMN way after two conv operations
        out = torch.cat((out, x), 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                                kernel_size=1, bias=False)

    def forward(x):
        out = self.conv1(F.relu(self.bn1(x)))
        # default padding=0 in avg_pool2d in Transition Layer
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


class DenseNet121(nn.Module):
    def __init__(self, in_channels, growth_rate, compression_rate, num_classes):
        super(DenseNet121, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels=in_channels)

        """
        Note: only when padding=3, the size of input image (224x224) can be coverted 
        to 112x112 according to the computing equation, since kernel_size(=7x7) and stride(=2) are fixed

        I also refer to original implementation: 
            https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2*growth_rate, kernel_size=7, \
                                stride=2, padding=3)

        """
        For each dense block, it consists of L bottlenecks.
        The number of output feature maps of a dense block = L*growth_rate + in_channels
        """
        self.dense_block1 = _make_dense_block(2*growth_rate, growth_rate, 6)
        channels_within_denseblock1 = 2*growth_rate + 6*growth_rate
        transition1_out_channels = math.floor(channels_within_denseblock1 * compression_rate)
        self.trans1 = Transition(channels_within_denseblock1, transition1_out_channels)

        self.dense_block2 = _make_dense_block(transition1_out_channels, growth_rate, 12)
        channels_within_denseblock2 = transition1_out_channels + 12*growth_rate
        transition2_out_channels =math.floor(channels_within_denseblock2 * compression_rate)
        self.trans2 = Transition(channels_within_denseblock2, transition2_out_channels)

        self.dense_block3 = _make_dense_block(transition2_out_channels, growth_rate, 24)
        channels_within_denseblock3 = transition2_out_channels + 24*growth_rate
        transition3_out_channels = math.floor(channels_within_denseblock3 * compression_rate)
        self.trans3 = Transition(channels_within_denseblock3, transition3_out_channels)

        """
        there is no Transition Layer 4 in DenseNet121 in their paper, Table 1
        HOWEVER, the official code add this Transition Layer, it is weird.
        """
        self.dense_block4 = _make_dense_block(transition3_out_channels, growth_rate, 16)
        channels_within_denseblock4 = transition3_out_channels + 16*growth_rate
        transition4_out_channels = math.floor(channels_within_denseblock4 * compression_rate)
        self.trans3 = Transition(channels_within_denseblock4, transition4_out_channels)

        """
        the first dim of self.fc() equals to transition4_out_channels,
        since after final global average pooling(7x7), the 7x7 feature map will be 1x1
        """
        self.fc = nn.Linear(transition4_out_channels, num_classes)

    # For DenseNet121, each dense block consists of different number of Bottleneck Layers
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        """
        @param: in_channels - the number of channels of input feature maps to a DenseBlock
        @param: growth_rate - the number of channels of output feature maps of a DenseBlock
        @param: num_layers - the number of Bottleneck Layers in a DenseBlock
        """
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels, growth_rate))
            # in_channels increases by 'growth_rate' after each Bottleneck
            in_channels += growth_rate

        return nn.sequential(*layers)

    def forward(x):
        out = self.conv1(F.relu(self.bn1(x)))
        # to keep feature map 56x56, here padding=1 in max_pool2d
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        out = self.dense_block1(out)
        out = self.trans1(out)

        out = self.dense_block2(out)
        out = self.trans2(out)

        out = self.dense_block3(out)
        out = self.trans3(out)
        
        # there is no Transition Layer 4 in DenseNet121
        out = self.dense_block4(out)

        out = F.relu(self.bn1(out))
        out = torch.squeeze(F.avg_pool2d(out, kernel_size=7))
        out = F.log_softmax(self.fc(out))
