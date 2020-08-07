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
    target_layer_num = 21
    step_size = 0.08
    clip = True
    octave_n = 5
    octave_scale = 1.8

    image = deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip)

    plt.figure()
    plt.imshow(image)
    plt.show()
