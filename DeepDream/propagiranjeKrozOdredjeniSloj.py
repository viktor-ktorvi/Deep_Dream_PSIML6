import torch
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
import requests
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

normalizeMean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
normalizeStd = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def loadModel():
    model = models.vgg16(pretrained=True).to(device)
    return model.eval()


def loadImage(filename):
    image = Image.open(filename)
    return np.array(image)


def prepareImage(image):
    image = np.float32(image)
    image /= 255.0
    return (image - normalizeMean) / normalizeStd


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loadModel()

    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    response = requests.get(LABELS_URL)
    labels = {int(key): value for key, value in response.json().items()}

    image = loadImage('data/input_images/pas.jpeg')
    image = prepareImage(image)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)


    imgTensor = torch.tensor(image[np.newaxis, :], requires_grad=True).to(device)
    imgTensor.retain_grad()

    prev_layer_output = imgTensor
    for i, layer in enumerate(model.features):
        print(i, type(layer))
        prev_layer_output = layer(prev_layer_output)
        print(prev_layer_output.shape)

    prev_layer_output.mean().backward()
    print(imgTensor.grad.shape)
    print(imgTensor.grad)

    imgGrad = imgTensor.grad.to("cpu").numpy()
    imgGrad = imgGrad[0]
    print(imgGrad.shape)
    # rgb,h,w --> h,w,rgb
    imgGrad = np.swapaxes(imgGrad, 0, 2)
    imgGrad = np.swapaxes(imgGrad, 0, 1)

    print(imgGrad.shape)
    plt.imshow(imgGrad)
    plt.show()



    print('dummy')