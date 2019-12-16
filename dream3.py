import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.resnet import resnet50, ResNet, Bottleneck
from utils import dream
import cv2 as cv
import time
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import numpy as np

epoch = 100
lr = 1.5
max_jitter = 32

noise = np.random.uniform(size=(1, 3, 512, 512)) + 100.0

input_tensor = Variable(torch.FloatTensor(noise), requires_grad=True)
# print(input_tensor)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class restnet(ResNet):
    def forward(self, x):
        """
        end_layer range from 1 to 4
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x[0][0]


# from models import resnet50

model = restnet(Bottleneck, [3, 4, 6, 3])
model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
if torch.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False
# optmizer = torch.optim.SGD([input_tensor], lr=0.01)

for i in range(100):
    # shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
    # input_tensor.data = np.roll(np.roll(input_tensor.data, shift_x, -1), shift_y, -2)
    model.zero_grad()
    out = model(input_tensor)
    print(i, out.shape, torch.sum(out))
    out.backward()
    ratio = np.abs(input_tensor.grad.data.cpu().numpy()).mean()
    learning_rate_use = lr / ratio
    print(ratio, learning_rate_use)
    input_tensor.data.add_(input_tensor.grad.data * learning_rate_use)
    # optmizer.step()
    input_tensor.grad.detach_()
    input_tensor.grad.zero_()
    time.sleep(1)
    if i % 10 == 0:
        img_tensor = input_tensor.squeeze().permute(1, 2, 0)
        image = Image.fromarray(np.uint8(img_tensor.detach().numpy()))
        image.show()
        image.save('data/cloud_{}.jpg'.format(i))
