import torch
from torch import nn
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class CustomResNet(models.resnet.ResNet):
    def forward(self, x, end_layer):
        """
        end_layer range from 1 to 4
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        if end_layer < 4:
            for i in range(end_layer):
                x = layers[i](x)
        else:
            for i in range(4):
                x = layers[i](x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet50(pretrained=True, **kwargs):
    model = CustomResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
