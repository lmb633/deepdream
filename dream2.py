import torch
from torchvision import transforms
from PIL import Image
from models import resnet50
import cv2 as cv
import time
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import numpy as np

epoch = 100
lr = 0.03
max_jitter = 32
img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_img = Image.open('data/timg.jpg')
input_tensor = img_transform(input_img)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = Variable(input_tensor, requires_grad=True)

model = resnet50()
if torch.cuda.is_available():
    model = model.cuda()
for param in model.parameters():
    param.requires_grad = False
# optmizer = torch.optim.SGD([input_tensor], lr=0.01)

for i in range(100):
    # shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
    # input_tensor.data = np.roll(np.roll(input_tensor.data, shift_x, -1), shift_y, -2)
    model.zero_grad()
    out = model(input_tensor, 3)
    print(i, out.shape, torch.sum(out))
    out.backward(out.data)
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
        print(img_tensor.shape)
        std = torch.Tensor([0.229, 0.224, 0.225])
        mean = torch.Tensor([0.485, 0.456, 0.406])
        img_tensor = img_tensor * std + mean
        img_tensor = np.clip(img_tensor.detach().numpy(), 0, 1)
        image = Image.fromarray(np.uint8(img_tensor * 255))
        image.show()
        image.save('tmp/cloud_{}.jpg'.format(i))
