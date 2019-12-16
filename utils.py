import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import numpy as np
import numpy as np
import torch
import scipy.ndimage as nd
from torch.autograd import Variable
import cv2 as cv

img_idx = 0


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    img = PIL.Image.fromarray(a)
    global img_idx
    img.save('tmp/cloud_{0}.jpg'.format(img_idx))
    img_idx += 1
    # img.show()


def showtensor(a):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    showarray(inp)
    clear_output(wait=True)


def objective_L2(dst, guide_features):
    return dst.data


def make_step(img, model, control=None, distance=objective_L2):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = 5e-2
    max_jitter = 32
    num_iterations = 50
    show_every = 10
    end_layer = 5
    guide_features = control
    out_channel = 275  # 猎犬

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        # print(shift_x, shift_y)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        else:
            img_variable = Variable(img_tensor, requires_grad=True)
        act_value = model.forward(img_variable, end_layer)
        if end_layer >= 4:
            act_value = act_value[0]
            # out_channel = np.argmax(act_value.detach().numpy())
            # print(out_channel)
            act_value = act_value[out_channel]
            print(act_value)
        diff_out = distance(act_value, guide_features)
        if diff_out < 0:
            diff_out = -diff_out
        act_value.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        print(learning_rate, ratio, learning_rate_use)
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        # print(img_variable.grad.data )
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std, (1 - mean) / std)
        if i == 0 or (i + 1) % show_every == 0:
            showtensor(img)
    return img


def dream(model, base_img, octave_n=6, octave_scale=1.4, control=None, distance=objective_L2):
    octaves = [base_img]
    print(octaves[0].shape)
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))
    detail = np.zeros_like(octaves[-1])

    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)
        input_oct = octave_base + detail
        out = make_step(input_oct, model, control, distance=distance)
        detail = out - octave_base
