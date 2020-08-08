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

    image = load_image('data/input_images/Maskenbal2018.jpg')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    octaves = create_octaves(image, 1.4, )
    octave = octaves[-1]
    print(np.min(octave), np.max(octave))
    imgTensor = torch.tensor(octave, requires_grad=True).to(device)
    imgTensor.retain_grad()

    # HERE DO JITTER!

    iter_n = 10
    target_layer_num = 15
    step_size = 0.2
    clip = True
    # for j in range(iter_n):
    #     target_layer_output = imgTensor
    #     # forward to target layer
    #     for i in range(target_layer_num):
    #         target_layer_output = model.features[i](target_layer_output)
    #
    #     model.zero_grad()
    #     target_layer_output.norm().backward()
    #     gradient = imgTensor.grad
    #
    #     imgTensor += step_size * gradient / torch.mean(torch.abs(gradient))
    #
    # imgGrad = imgTensor.to("cpu").detach().numpy()
    #
    # clip = False
    # # clipping
    # if clip:
    #     mini = -normalizeMean / normalizeStd
    #     maxi = (1 - normalizeMean) / normalizeStd
    #     for i in range(3):
    #         imgGrad[0, i, :] = np.clip(imgGrad[0, i, :], mini[i], maxi[i])

    imgGrad = octave
    for j in range(iter_n):
        imgGrad = next_step(imgGrad, model, target_layer_num, step_size, clip)

    # imgGrad = imgTensor.to("cpu").detach().numpy()
    imgGrad = deprocess(imgGrad)

    depOct = deprocess(octave)
    details = depOct - imgGrad
    plt.figure()
    plt.imshow(imgGrad)

    plt.figure()
    plt.imshow(details)

    plt.figure()
    plt.imshow(depOct)
    print(np.min(depOct), np.max(depOct))

    plt.show()
