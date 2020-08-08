from utils import *
from scipy.ndimage import gaussian_filter

import torch.nn as nn


def gaussian(M, std, sym=True):

    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]

    print(w)
    print(w.shape)
    return w


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    iter_n = 10
    target_layer_num = 16
    step_size = 0.04
    clip = True
    octave_n = 5
    octave_scale = 1.8

    # SLIKA
    filename = "Maskenbal2018"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = load_image(filepath + filename + extension)

    plt.figure()
    plt.imshow(image)

    img_tensor = preprocess(image, device)

    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=264, bias=False)
    with torch.no_grad():
        conv.weight = nn.Parameter(gaussian(5, 1))

    img_tensor = conv(img_tensor)

    image = deprocess(img_tensor)
    plt.figure()
    plt.imshow(image)

    plt.show()
