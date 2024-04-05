import torch
from torch import nn


class Bottleneck(nn.Module):
    extension = 4

    def __init__(self, inplanes, planes, stride, downsample=None):
        '''
        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.extension, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.extension)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class):
        super(ResNet, self).__init__()
        self.inplane = 64
        self.block = block
        self.layers = layers

        self.conv1 = nn.Conv3d(1, self.inplane, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm3d(self.inplane)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.stage1 = self.make_layer(block, 64, self.layers[0], stride=1)
        self.stage2 = self.make_layer(block, 256, self.layers[1], stride=2)
        self.stage3 = self.make_layer(block, 128, self.layers[2], stride=2)
        self.stage4 = self.make_layer(block, 512, self.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.maxpool(out)
        out = self.relu(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def make_layer(self, block, plane, block_num, stride=1):
        '''
            :param block: block模板
            :param plane: 每个模块中间运算的维度，一般等于输出维度/4
            :param block_num: 重复次数
            :param stride: 步长
            :return:
        '''
        block_list = []
        downsample = None
        if (stride != 1 or self.inplane != plane * block.extension):
            downsample = nn.Sequential(
                nn.Conv3d(self.inplane, plane * block.extension, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm3d(plane * block.extension)
            )
        conv_block = block(self.inplane, plane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = plane * block.extension

        for i in range(1, block_num):
            block_list.append(block(self.inplane, plane, stride=1))

        return nn.Sequential(*block_list)


if __name__ == '__main__':
    resnet = ResNet(Bottleneck, [1, 1, 1, 1], 100)
    x = torch.rand(2, 1, 128, 128, 128)
    X = resnet(x)
    print(X.shape)