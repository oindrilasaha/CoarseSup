import torch.nn as nn
import torch
import math
from torch import clamp 
import torch.utils.model_zoo as model_zoo
import torchvision

__all__ = ['fcn_posterior', 'UpBlock']

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            #x = nn.functional.interpolate(x, 
            #    scale_factor=2, mode="bilinear", align_corners=True)
            # x = nn.functional.unsample_bilinear(x, scale_factor=2)
            x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, nparts, input_dim):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)

        self.model_image = torchvision.models.resnet34()
        self.inplanes=256
        self.model_image.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.model_image.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=True)


        self.decoder = nn.Sequential(
                UpBlock(528, 256, upsample=True),
                UpBlock(256, 128, upsample=True),
                UpBlock(128, 64, upsample=True),
                UpBlock(64, 64, upsample=True),
                nn.Conv2d(64, nparts, 1, 1),
                )
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        x_img = x[:,0:3]
        x = x[:,3:]
        # import pdb;pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # with torch.no_grad():
        z_img = self.model_image.conv1(x_img)
        z_img = self.model_image.bn1(z_img)
        z_img = self.model_image.relu(z_img)
        z_img = self.model_image.maxpool(z_img)
        z_img = self.model_image.layer1(z_img)
        z_img = self.model_image.layer2(z_img)
        z_img = self.model_image.layer3(z_img)
        z_img = self.model_image.layer4(z_img)

        z = self.decoder(torch.cat([x, z_img],1))

        return z

def fcn_posterior(input_dim=3, pretrained=False, nparts=15):
    model = ResNet(BasicBlock, [1, 1, 6, 3], nparts=nparts, input_dim=input_dim)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)

    return model
