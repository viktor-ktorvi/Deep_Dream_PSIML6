import torch
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.models as models
import requests
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from utils import *


def predictVGG16(filename, topN=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    response = requests.get(LABELS_URL)
    labels = {int(key): value for key, value in response.json().items()}

    image = load_image(filename)
    image_tensor = preprocess(image, "cuda")

    with torch.no_grad():
        prediction = model(image_tensor)

    # listOfPredisctions = prediction.tolist()
    # listOfProbabilities = torch.nn.functional.softmax(prediction, dim=0)

    prediction = torch.nn.functional.softmax(prediction.data.numpy(), dim=1)

    indexes = np.argsort(prediction)

    print("N\t", "Score\t\t", "Class\n")
    for i in range(1, topN):
        print(i, "\t", prediction[0, indexes[0, -i]], "\t", labels[indexes[0, -i]])


if __name__ == "__main__":
    predictVGG16('data/input_images/MaskenbalViktor.jpeg', 10)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_model()
    #
    # LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    # response = requests.get(LABELS_URL)
    # labels = {int(key): value for key, value in response.json().items()}
    #
    # image = load_image('data/input_images/Maskenbal2018.jpg')
    # image = preprocess(image)
    # # h,w,rgb --> rgb,h,w
    # image = np.swapaxes(image, 0, 2)
    # image = np.swapaxes(image, 1, 2)
    #
    # imgTensor = torch.from_numpy(image[np.newaxis, :]).to(device)
    # with torch.no_grad():
    #     prediction = model(imgTensor).to("cpu")
    #
    # # listOfPredisctions = prediction.tolist()
    # # listOfProbabilities = torch.nn.functional.softmax(prediction, dim=0)
    #
    # prediction = prediction.data.numpy()
    #
    # indexes = np.argsort(prediction)
    #
    #
    # for i in range(1,10):
    #     print(i, prediction[0, indexes[0, -i]], labels[indexes[0, -i]])
