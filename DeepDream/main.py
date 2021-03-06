import os
from utils import *
from datetime import datetime
import cv2
from matplotlib import pyplot as plt
import time

if __name__ == "__main__":
    start_time = time.time()

    # NOISE IF YOU WANT IT
    n = 1200
    gauss_mean = 255.0 / 2.0
    sigma = 255.0 / 6
    img = gauss_mean + sigma * np.random.randn(n, n, 3)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    noise_img = img

    # IMAGE
    filename = "lamas"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = load_image(filepath + filename + extension)

    # GUIDE
    guide_filename = "monarch"
    guide_extension = ".jpg"
    guide = load_image(filepath + guide_filename + guide_extension)

    # za guided:
    # guide (150, 100) otp, glavna slika (700, 700) super rezultate daje
    # veliki broj oktava je dobar(10), oct_scale(1.4), sto vise iteracija to jasniji guide

    # PARAMETERS
    iter_n = 10
    target_layer_num = 16   # if not making a gif

    step_size = 0.02
    clip = True
    octave_n = 10
    octave_scale = 1.4
    guided = True
    blur = True

    if guided:
        iter_n = 20

    # RESCALING
    guide_resize = 150
    if guide.shape[0] > guide.shape[1]:
        guide_scale = guide.shape[0] / guide_resize
    else:
        guide_scale = guide.shape[1] / guide_resize

    h = round(guide.shape[0] / guide_scale)
    w = round(guide.shape[1] / guide_scale)
    guide = cv2.resize(guide, (w, h))

    ###

    img_resize = 700
    if image.shape[0] > image.shape[1]:
        scale = image.shape[0] / img_resize
    else:
        scale = image.shape[1] / img_resize

    h = round(image.shape[0] / scale)
    w = round(image.shape[1] / scale)
    image = cv2.resize(image, (w, h))

    plt.figure()
    plt.imshow(image)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = load_model(device)

    # FILE NAMING
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
    # info += "Target layer number:" + tabs + str(target_layer_num) + "\n"
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

    # THE MAKING OF
    image_copy = np.copy(image)
    image_list = [Image.fromarray(image)]
    for i in range(len(model.features)):
        image = deep_dream(image_copy, model, device, octave_n, octave_scale, iter_n, i, step_size, clip,
                           guided=guided, guide=guide, blur=blur)
        im = Image.fromarray(image)
        image_list.append(im)
        im.save(path + "/" + filename + str(i) + ".jpg")
        print(i)

    image_list[0].save(path + "/" + filename + ".gif",
                       save_all=True, append_images=image_list[:], optimize=False, duration=500, loop=0)

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.show()
