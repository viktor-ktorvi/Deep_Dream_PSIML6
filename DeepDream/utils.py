import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
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
    return image


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
        # DEBAG PRINTOVI
        # print("h i w  = ", h, w)
        # risajz = np.resize(octaves[-1], (h, w, 3))
        #
        # print ("Risajz\n", risajz)
        img = cv2.resize(octaves[-1], (w, h), interpolation=cv2.INTER_CUBIC)
        octaves.append(img)


    if normalized:
        for i in range(len(octaves)):
            octaves[i] = prepareImage(octaves[i])

    return octaves
