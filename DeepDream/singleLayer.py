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
    filename = "adam"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = load_image(filepath + filename + extension)

    guide_filename = "monarch"
    guide_extension = ".jpg"
    guide = load_image(filepath + guide_filename + guide_extension)

    guide_scale = guide.shape[0] / 150
    h = round(guide.shape[0] / guide_scale)
    w = round(guide.shape[1] / guide_scale)
    guide = cv2.resize(guide, (w, h))

    size_resize = 400
    if image.shape[0] > image.shape[1]:
        scale = image.shape[0] / size_resize
    else:
        scale = image.shape[1] / size_resize

    # scale = image.shape[0] / size_resize
    h = round(image.shape[0] / scale)
    w = round(image.shape[1] / scale)
    # image = cv2.resize(image, (w, h))

    # filename = "noise"
    # image = noise_img

    plt.figure()
    plt.imshow(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # za guided:
    # guide (150, 100) otp, glavna slika (700, 700) super rezultate daje
    # veliki broj oktava je dobar(10), oct_scale(1.4), sto vise iteracija to jasniji guide

    iter_n = 10
    target_layer_num = 28
    step_size = 0.02
    clip = True
    octave_n = 8
    octave_scale = 1.4
    guided = False
    blur = True

    if guided:
        iter_n = 20

    # print(type(model.features[target_layer_num - 1]))
    # image = deep_dream(image, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip,
    #                    guided=guided, guide=guide, blur=blur)
    # plt.figure()
    # plt.imshow(image)

    dateTimeObj = datetime.now()
    vreme = "_time_" + str(dateTimeObj.hour) + 'h_' + str(dateTimeObj.minute) + 'm_' + str(dateTimeObj.second) + "s"
    location = "data/gif/"
    path = location + filename + vreme
    if guided:
        path += "_guided"
    os.makedirs(path)
    text_file = open(path + "/info.txt", "w")
    tabs = ""
    tab_num = 1
    for i in range(tab_num):
        tabs += "\t"
    info = "Name:" + tabs + filename + extension + "\n"
    info += "(h, w):" + tabs + str(image.shape) + "\n"
    info += "Number of iterations:" + tabs + str(iter_n) + "\n"
    # info += "Taregt layer number:" + tabs + str(target_layer_num) + "\n"
    # info += "Layer info:" + tabs + str(type(model.features[target_layer_num - 1])) + "\n"
    info += "Step size:" + tabs + str(step_size) + "\n"
    info += "Clipping:" + tabs + str(clip) + "\n"
    info += "Number of octaves:" + tabs + str(octave_n) + "\n"
    info += "Octave scale:" + tabs + str(octave_scale) + "\n"
    info += "Guided:" + tabs + str(guided) + "\n"
    if guided:
        info += "Guide image name:" + tabs + guide_filename + guide_extension + "\n"
        info += "Guide (h, w):" + tabs + str(guide.shape) + "\n"
    info += "Blur:" + tabs + str(blur) + "\n"
    print(info)
    text_file.write(info)
    text_file.close()

    if guided:
        im = Image.fromarray(guide)
        im.save(path + "/" + "guide" + ".jpg")

    image_copy = np.copy(image)
    image_list = [Image.fromarray(image)]
    for i in [23, 24, 25, 26]:
        image = deep_dream(image_copy, model, device, octave_n, octave_scale, iter_n, i, step_size, clip,
                           guided=guided, guide=guide, blur=blur)
        im = Image.fromarray(image)
        image_list.append(im)
        im.save(path + "/" + filename + str(i) + ".jpg")
        print(i)

    image_list[0].save(path + "/" + filename + ".gif",
                       save_all=True, append_images=image_list[0:], optimize=False, duration=250, loop=0)

    plt.show()
