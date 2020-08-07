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

image = loadImage("data/input_images/pas.jpeg")
plt.figure()
plt.imshow(image)

print("Original Image\n", image)
octaves = createOctaves(image, 4, 1.4, normalized=False)
img = octaves[-1]
print("ORIGINAL OKTAVA")
print(img)
plt.figure()
plt.imshow(img)
# img = prepareImage(img)
# print("POSLE PREPARE OKTAVA")
# print(img)
# img = deprocess(img[np.newaxis, :])
# print("POSLE DEPROCESS OKTAVA")
#
# print(img)
# img = img.astype(np.uint8)
# print("POSLE ASTYPE OKTAVA")
#
# print(img)

plt.show()