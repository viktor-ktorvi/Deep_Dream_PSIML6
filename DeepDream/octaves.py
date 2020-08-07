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

    image = loadImage('data/input_images/pas.jpeg')
    octaves = createOctaves(image, 4, 1.4)
    for i in octaves:
        print(i.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loadModel(device)

    imgTensor = torch.tensor(octaves[-1][np.newaxis, :], requires_grad=True).to(device)
    imgTensor.retain_grad()

    # HERE DO JITTER!

    target_layer_output = imgTensor

    klon = imgTensor.clone()

    target_layer_num = 25
    # forward to target layer
    for i in range(target_layer_num):
        print(i, type(model.features[i]))
        target_layer_output = model.features[i](target_layer_output)

    # criteria is mean, lets maximize mean
    model.zero_grad()
    target_layer_output.mean().backward()
    gradient = imgTensor.grad

    step_size = 0.002

    imgTensor += step_size * gradient / torch.mean(torch.abs(gradient))

    justGrad = step_size * gradient / torch.mean(torch.abs(gradient))
    print(justGrad)

    imgGrad = imgTensor.to("cpu").detach().numpy()
    justGrad = justGrad.to("cpu").detach().numpy()
    klon = klon.to("cpu").detach().numpy()

    justGrad = deprocess(justGrad)
    imgGrad = deprocess(imgGrad)
    klon = deprocess(klon)

    plt.figure()
    plt.imshow(imgGrad)

    plt.figure()
    plt.imshow(justGrad)

    # print(imgGrad)
    # print(imgGrad.shape)


    plt.figure()
    plt.imshow(klon)
    plt.show()

