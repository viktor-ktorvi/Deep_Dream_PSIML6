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
from datetime import datetime

if __name__ == "__main__":
    # ABGSH
    n = 1200
    gauss_mean = 255.0 / 2.0
    sigma = 255.0 / 6
    img = gauss_mean + sigma * np.random.randn(n, n, 3)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    noise_img = img

    # SLIKA
    filename = "clouds"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = load_image(filepath + filename + extension)

    guide_filename = "monarch"
    guide_extension = ".jpg"
    guide = load_image(filepath + guide_filename + guide_extension)

    guide_scale = 2
    h = round(guide.shape[0] / guide_scale)
    w = round(guide.shape[1] / guide_scale)
    guide = cv2.resize(guide, (w, h))

    scale = 2
    h = round(image.shape[0] / scale)
    w = round(image.shape[1] / scale)
    image = cv2.resize(image, (w, h))

    # filename = "noise"
    # image = noise_img

    plt.figure()
    plt.imshow(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # za guided:
    # guide (150, 100) otp, glavna slika (700, 700) super rezultate daje
    # veliki broj oktava je dobar(10), oct_scale(1.4), sto vise iteracija to jasniji guide

    iter_n = 20
    target_layer_num = 28
    step_size = 0.01
    clip = True
    octave_n = 10
    octave_scale = 1.4
    guided = True
    blur = True

    print(type(model.features[target_layer_num - 1]))
    image = deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip,
                       guided=guided, guide=guide, blur=blur)
    plt.figure()
    plt.imshow(image)

    im = Image.fromarray(image)
    outpath = "data/fun_figures/"
    dateTimeObj = datetime.now()
    vreme = str(dateTimeObj.hour) + '_' + str(dateTimeObj.minute) + '_' + str(dateTimeObj.second)
    im.save(outpath + filename + "_iter_n_" + str(iter_n) + "_layer_" + str(target_layer_num) + "_step_" + str(
        step_size) + "_octave_n_" + str(octave_n) + "_octave_scale_" + str(octave_scale) + "_blur_" + str(
        blur) + "_vreme_" + vreme + ".jpg")

    plt.show()
