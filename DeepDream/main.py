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
    n = 1200
    img = 255.0 / 2.0 + 255.0 / 6 * np.random.randn(n, n, 3)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    noise_img = img

    # SLIKA
    filename = "clouds"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = load_image(filepath + filename + extension)

    guide = load_image(filepath + "owl2.jpg")
    scale = 3
    h = round(guide.shape[0] / scale)
    w = round(guide.shape[1] / scale)
    guide = cv2.resize(guide, (w, h))

    scale = 2.5
    h = round(image.shape[0] / scale)
    w = round(image.shape[1] / scale)
    #image = cv2.resize(image, (w, h))

    # filename = "noise"
    # image = noise_img

    plt.figure()
    plt.imshow(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)

    iter_n = 15
    target_layer_num = 29
    step_size = 0.04
    clip = True
    octave_n = 5
    octave_scale = 1.8
    guided = True

    image = deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip, guided, guide)
    plt.figure()
    plt.imshow(image)

    #cat
    # img_copy = image
    # for i in range(14,len(model.features)):
    #     image = deep_dream(img_copy, model, device, octave_n, octave_scale, iter_n, i, step_size, clip)
    #     print(i)
    #     im = Image.fromarray(image)
    #     outpath = "data/cats/"
    #     im.save(outpath + filename + "_iter_n_" + str(iter_n) + "_layer_" + str(i) + "_step_" + str(
    #         step_size) + "_octave_n_" + str(octave_n) + "_octave_scale_" + str(octave_scale) + ".jpg")

    im = Image.fromarray(image)
    outpath = "data/fun_figures/"
    im.save(outpath + filename + "_iter_n_" + str(iter_n) + "_layer_" + str(target_layer_num) + "_step_" + str(step_size) + "_octave_n_" + str(octave_n) + "_octave_scale_" + str(octave_scale) +".jpg")

    plt.show()


