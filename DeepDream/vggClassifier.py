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

    image = loadImage('data/input_images/Maskenbal2018.jpg')
    image = prepareImage(image)
    # h,w,rgb --> rgb,h,w
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)

    imgTensor = torch.from_numpy(image[np.newaxis, :]).to(device)
    with torch.no_grad():
        prediction = model(imgTensor).to("cpu")

    # listOfPredisctions = prediction.tolist()
    # listOfProbabilities = torch.nn.functional.softmax(prediction, dim=0)

    prediction = prediction.data.numpy()

    indexes = np.argsort(prediction)


    for i in range(1,10):
        print(i, prediction[0, indexes[0, -i]], labels[indexes[0, -i]])

