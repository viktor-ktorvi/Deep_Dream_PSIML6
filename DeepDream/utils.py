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
    return torch.tensor(image[np.newaxis, :], requires_grad=True, device=device)


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


def next_step(img_tensor, model, target_layer_num=1, step_size=0.02, clip=True, guided=False, guide_features=None,
              blur=True, sigma=1):

    img_tensor.retain_grad()

    # forward img to target layer
    target_layer_output = propagate(img_tensor, model, target_layer_num)
    model.zero_grad()

    if guided:
        guided_gram = gram_matrix(guide_features)
        target_gram = gram_matrix(target_layer_output)
        (guided_gram - target_gram).norm().backward(retain_graph=True)
    else:
        target_layer_output.norm().backward()

    gradient = img_tensor.grad
    if blur:
        gradient = gradient.cpu().detach().numpy()
        # (gradient, sigma=(0.0, 0.0, sigma, sigma))
        grad_smooth1 = gaussian_filter(gradient, sigma=(0.0, 0.0, sigma, sigma))
        grad_smooth2 = gaussian_filter(gradient, sigma=(0.0, 0.0, sigma*2, sigma*2))
        grad_smooth3 = gaussian_filter(gradient, sigma=(0.0, 0.0, sigma*0.5, sigma*0.5))

        # grad_smooth1 = gaussian_filter(gradient, sigma=sigma)
        # grad_smooth2 = gaussian_filter(gradient, sigma=sigma * 2)
        # grad_smooth3 = gaussian_filter(gradient, sigma=sigma * 0.5)

        gradient = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        gradient = gaussian_filter(gradient, sigma)
        gradient = torch.tensor(gradient).to("cuda")

    if guided:
        img_tensor -= step_size * gradient / torch.mean(torch.abs(gradient))
    else:
        img_tensor += step_size * gradient / torch.mean(torch.abs(gradient))

    if clip:
        mini = -normalizeMean / normalizeStd
        maxi = (1 - normalizeMean) / normalizeStd
        for i in range(3):
            img_tensor.data[0, i, :] = torch.clamp(img_tensor.data[0, i, :], mini[i], maxi[i])

    return img_tensor


def deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip, guided=False,
               guide=None, blur=True):
    img_tensor = preprocess(image, device)

    if guided:
        guide = preprocess(guide, device)
        guided_features = propagate(guide, model, target_layer_num)
    else:
        guided_features = None

    octaves = create_octaves(img_tensor, octave_n, octave_scale)

    detail = torch.zeros_like(octaves[-1], requires_grad=True)

    octaves = octaves[::-1]
    for i, octave in enumerate(octaves):
        h, w = octave.shape[-2:]

        if i > 0:
            h_next, w_next = octaves[i].shape[-2:]
            detail = resize_tensor(detail, h_next, w_next)

        img_tensor = resize_tensor(img_tensor, h, w)
        img_tensor = octave + detail

        for j in range(iter_n):
            sigma = (i * 4.0) / iter_n + 0.5
            sigma = 1
            img_tensor = next_step(img_tensor, model, target_layer_num, step_size, clip, guided, guided_features, blur,
                                   sigma)

        detail = img_tensor - octave

    return deprocess(img_tensor)


def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input_tensor.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
