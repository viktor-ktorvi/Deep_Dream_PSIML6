import torch
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
import requests
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from skimage.transform import resize
from utils import *

if __name__ == "__main__":

    # va visim slojevima
    # kada radimo sa abgs slikom kao da levi gornji cosak uvek pojacava a donji desni nikad! Pitati aleksu

    # ABGSH
    n = 600
    img = 255.0 / 2.0 + 255.0 / 6 * np.random.randn(n, n, 3)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    print(np.max(img), np.min(img))
    print(img)
    image = img

    # SLIKA
    image = loadImage('data/input_images/flamingo.jpg')
    image = cv2.resize(image, (1000, 1000))

    plt.figure()
    plt.imshow(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loadModel(device)

    iter_n = 6
    # sloj 30 za ptice
    target_layer_num = 25
    step_size = 0.08
    clip = True
    octave_n = 5
    octave_scale = 1.8

    octaves = createOctaves(image, octave_n, octave_scale, normalized=True)
    image = prepareImage(image)

    detail = np.zeros_like(octaves[-1])

    octaves = octaves[::-1]
    for i, octave in enumerate(octaves):
        # print(octave.shape)
        h, w = octave.shape[-2:]
        print(octave.shape[-2:])

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

        print("Image:\n", image.shape)
        print("Octave\n", octave.shape)
        print("Detail\n", detail.shape)
        image = octave + detail

        for j in range(iter_n):
            image = next_step(image, model, device, target_layer_num, step_size, clip)

        detail = image - octave

    image = deprocess(image)

    plt.figure()
    plt.imshow(image)
    plt.show()
