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

# ImageNet mean, training set dependent
normalizeMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
normalizeStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def loadModel(device):
    model = models.vgg16(pretrained=True).to(device)
    return model.eval()


def loadImage(filename):
    image = Image.open(filename)
    return np.array(image)


def prepareImage(image):
    image = np.float32(image)
    image /= 255.0
    image = (image - normalizeMean) / normalizeStd
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image[np.newaxis, :]


def deprocess(image):
    image = image[0]
    # rgb,h,w --> h,w,rgb
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    image *= normalizeStd
    image += normalizeMean
    image = np.clip(image, 0., 1.)

    image *= 255
    return image.astype(np.uint8)


def createOctaves(image, octave_n, octave_scale, normalized=True):
    octaves = [image]
    for i in range(octave_n - 1):
        h = np.int(octaves[-1].shape[0] / octave_scale)
        w = np.int(octaves[-1].shape[1] / octave_scale)

        img = cv2.resize(octaves[-1], (w, h), interpolation=cv2.INTER_CUBIC)
        octaves.append(img)

    if normalized:
        for i in range(len(octaves)):
            octaves[i] = prepareImage(octaves[i])

    return octaves


def propagate(img_tensor, model, target_layer_num):
    target_layer_output = img_tensor
    # forward to target layer
    for i in range(target_layer_num):
        target_layer_output = model.features[i](target_layer_output)

    return target_layer_output


def next_step(img, model, device, target_layer_num=1, step_size=0.02, clip=True, guided = False):
    # img_tensor : current tensor to ascent
    # model : chosen model
    # target_layer_num : order of layer in model (1..end)
    # step_size : learning rate
    # clip : saturation of image pixels, makes sure that in the end they are 0..255
    # init with current image
    img_tensor = torch.tensor(img, requires_grad=True).to(device)
    img_tensor.retain_grad()

    # target_layer_output = img_tensor
    # # forward to target layer
    # for i in range(target_layer_num):
    #     target_layer_output = model.features[i](target_layer_output)

    # forward img to target layer
    target_layer_output = propagate(img_tensor, model, target_layer_num)

    model.zero_grad()
    if guided:
        pass
    else:
        target_layer_output.norm().backward()

    gradient = img_tensor.grad
    # sigma = 5
    # gradient = gradient.to("cpu")
    # grad_smooth1 = gaussian_filter(gradient, sigma=sigma)
    # grad_smooth2 = gaussian_filter(gradient, sigma=sigma * 2)
    # grad_smooth3 = gaussian_filter(gradient, sigma=sigma * 0.5)
    #
    # gradient = (grad_smooth1 + grad_smooth2 + grad_smooth3)
    # gradient = torch.tensor(gradient, requires_grad=False).to(device)

    # BLUR end

    img_tensor += step_size * gradient / torch.mean(torch.abs(gradient))

    img = img_tensor.to("cpu").detach().numpy()

    # clipping
    if clip:
        mini = -normalizeMean / normalizeStd
        maxi = (1 - normalizeMean) / normalizeStd
        for i in range(3):
            img[0, i, :] = np.clip(img[0, i, :], mini[i], maxi[i])

    return img

def next_step_copy(img, model, device, target_layer_num=1, step_size=0.02, clip=True, guided = False):
    # img_tensor : current tensor to ascent
    # model : chosen model
    # target_layer_num : order of layer in model (1..end)
    # step_size : learning rate
    # clip : saturation of image pixels, makes sure that in the end they are 0..255
    # init with current image
    img_tensor = torch.tensor(img, requires_grad=True).to(device)
    img_tensor.retain_grad()

    # target_layer_output = img_tensor
    # # forward to target layer
    # for i in range(target_layer_num):
    #     target_layer_output = model.features[i](target_layer_output)

    # forward img to target layer
    target_layer_output = propagate(img_tensor, model, target_layer_num)

    model.zero_grad()
    if guided:
        pass
    else:
        target_layer_output.norm().backward()

    gradient = img_tensor.grad

    img_tensor += step_size * gradient / torch.mean(torch.abs(gradient))

    img = img_tensor.to("cpu").detach().numpy()

    # clipping
    if clip:
        mini = -normalizeMean / normalizeStd
        maxi = (1 - normalizeMean) / normalizeStd
        for i in range(3):
            img[0, i, :] = np.clip(img[0, i, :], mini[i], maxi[i])

    return img


def deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip):
    octaves = createOctaves(image, octave_n, octave_scale, normalized=True)
    image = prepareImage(image)

    detail = np.zeros_like(octaves[-1])

    octaves = octaves[::-1]
    for i, octave in enumerate(octaves):
        # print(octave.shape)
        h, w = octave.shape[-2:]

        if i > 0:
            h_next, w_next = octaves[i].shape[-2:]
            detail = deprocess(detail)
            # print(detail.shape)
            detail = cv2.resize(detail, (w_next, h_next), interpolation=cv2.INTER_CUBIC)
            detail = prepareImage(detail)

        image = deprocess(image)
        # print(image.shape)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        image = prepareImage(image)

        image = octave + detail

        for j in range(iter_n):
            image = next_step(image, model, device, target_layer_num, step_size, clip)

        detail = image - octave

    return deprocess(image)
