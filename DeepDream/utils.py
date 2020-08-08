import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
from scipy.ndimage.filters import gaussian_filter
import requests
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from skimage.transform import resize

import torch.nn.functional as F
from torch.autograd import Variable

# ImageNet mean, training set dependent
normalizeMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
normalizeStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(device):
    model = models.vgg16(pretrained=True).to(device)
    return model.eval()


def load_image(filename):
    image = Image.open(filename)
    return np.array(image)


def preprocess(image, device):
    image = np.float32(image)
    image /= 255.0
    image = (image - normalizeMean) / normalizeStd
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    # mozda tensor = torch.tensor(image[np.newaxis, :], requires_grad=True)
    # tensor = tensor.to(device)
    # return tensor
    return torch.tensor(image[np.newaxis, :], requires_grad=True).to(device)


# treba promeniti
def deprocess(img_tensor):
    img_tensor = img_tensor[0]
    image = img_tensor.cpu().detach().numpy()
    # rgb,h,w --> h,w,rgb
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    image *= normalizeStd
    image += normalizeMean
    image = np.clip(image, 0., 1.)

    image *= 255
    return image.astype(np.uint8)


def create_octaves(img_tensor, octave_n, octave_scale):
    octaves = [img_tensor]
    for i in range(octave_n - 1):
        h = round(octaves[-1].shape[-2] / octave_scale)
        w = round(octaves[-1].shape[-1] / octave_scale)

        # reshape
        # new_octave = torch.tensor(F.adaptive_avg_pool2d(Variable(octaves[-1]), (h, w)), requires_grad=True)
        new_octave = resize_tensor(octaves[-1], h, w)
        octaves.append(new_octave)

    return octaves


def resize_tensor(original, h, w):
    return F.adaptive_avg_pool2d(Variable(original), (h, w)).clone().detach().requires_grad_(True)


def propagate(img_tensor, model, target_layer_num):
    target_layer_output = img_tensor
    # forward to target layer
    for i in range(target_layer_num):
        target_layer_output = model.features[i](target_layer_output)

    return target_layer_output


def next_step(img_tensor, model, target_layer_num=1, step_size=0.02, clip=True):
    # img_tensor : current tensor to ascent
    # model : chosen model
    # target_layer_num : order of layer in model (1..end)
    # step_size : learning rate
    # clip : saturation of image pixels, makes sure that in the end they are 0..255
    # init with current image
    img_tensor.retain_grad()

    # forward img to target layer
    target_layer_output = propagate(img_tensor, model, target_layer_num)
    torch.autograd.set_detect_anomaly(True)
    model.zero_grad()
    target_layer_output.norm().backward()
    gradient = img_tensor.grad

    # print(type(gradient), type(img_tensor))
    img_tensor += step_size * gradient / torch.mean(torch.abs(gradient))

    if clip:
        mini = -normalizeMean / normalizeStd
        maxi = (1 - normalizeMean) / normalizeStd
        for i in range(3):
            img_tensor.data[0, i, :] = torch.clamp(img_tensor.data[0, i, :], mini[i], maxi[i])

    return img_tensor


def deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip):
    img_tensor = preprocess(image, device)
    # img_tensor.retain_grad()
    octaves = create_octaves(img_tensor, octave_n, octave_scale)

    detail = torch.zeros_like(octaves[-1], requires_grad=True)

    octaves = octaves[::-1]
    for i, octave in enumerate(octaves):
        h, w = octave.shape[-2:]

        if i > 0:
            h_next, w_next = octaves[i].shape[-2:]
            # detail = (F.adaptive_avg_pool2d(Variable(detail), (h_next, w_next))).data
            detail = resize_tensor(detail, h_next, w_next)

        # img_tensor = (F.adaptive_avg_pool2d(Variable(img_tensor), (h, w))).data
        img_tensor = resize_tensor(img_tensor, h, w)
        img_tensor = octave + detail

        for j in range(iter_n):
            img_tensor = next_step(img_tensor, model, target_layer_num, step_size, clip)

        detail = img_tensor - octave

    return deprocess(img_tensor)
