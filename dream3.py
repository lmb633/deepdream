import torch
from torchvision import transforms
from PIL import Image
import cv2 as cv
import time
from torch.autograd import Variable
import numpy as np
from models import resnet50

epoch = 100
lr = 2.5
max_jitter = 32
out_class = 0

noise = np.random.uniform(size=(1, 3, 512, 512)) + 100.0
input_tensor = Variable(torch.FloatTensor(noise), requires_grad=True)
# print(input_tensor)


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
    out = model(input_tensor, 4)[0][out_class]
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
