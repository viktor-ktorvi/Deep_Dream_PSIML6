from utils import *
import torch.nn.functional as nnf


def next_step_test(img, guided_features, model, device, target_layer_num=1, step_size=0.02, clip=True, guided=True):
    # img_tensor : current tensor to ascent
    # model : chosen model
    # target_layer_num : order of layer in model (1..end)
    # step_size : learning rate
    # clip : saturation of image pixels, makes sure that in the end they are 0..255
    # init with current image
    img_tensor = torch.tensor(img, requires_grad=True).to(device)
    img_tensor.retain_grad()

    # target_layer_output = img_tensor
    # # forward to target layer
    # for i in range(target_layer_num):
    #     target_layer_output = model.features[i](target_layer_output)

    # forward img to target layer
    target_layer_output = propagate(img_tensor, model, target_layer_num)

    model.zero_grad()
    if guided:
        print(guided_features.shape)
        print(target_layer_output.shape)
        # target_layer_output = torch.reshape(target_layer_output, guided_features.shape)
        guided_features = nnf.interpolate(guided_features, size=target_layer_output.shape[-2:], mode='bicubic',
                                              align_corners=False)

        cost = (guided_features - target_layer_output).norm()
        cost.backward(retain_graph=True)
        gradient = img_tensor.grad
        img_tensor -= step_size * gradient / torch.mean(torch.abs(gradient))
    else:
        target_layer_output.norm().backward()
        gradient = img_tensor.grad
        img_tensor += step_size * gradient / torch.mean(torch.abs(gradient))

    img = img_tensor.to("cpu").detach().numpy()

    # clipping
    if clip:
        mini = -normalizeMean / normalizeStd
        maxi = (1 - normalizeMean) / normalizeStd
        for i in range(3):
            img[0, i, :] = np.clip(img[0, i, :], mini[i], maxi[i])

    return img


def deep_dream_test(image, guide, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size, clip):
    octaves = create_octaves(image, octave_scale, )
    image = preprocess(image, "cuda")

    guide = preprocess(guide, "cuda")
    guided_features = torch.tensor(guide, requires_grad=False).to(device)

    guided_features = propagate(guided_features, model, target_layer_num)
    detail = np.zeros_like(octaves[-1])

    octaves = octaves[::-1]
    for i, octave in enumerate(octaves):
        # print(octave.shape)
        h, w = octave.shape[-2:]

        if i > 0:
            h_next, w_next = octaves[i].shape[-2:]
            detail = deprocess(detail)
            # print(detail.shape)
            detail = cv2.resize(detail, (w_next, h_next), interpolation=cv2.INTER_CUBIC)
            detail = preprocess(detail, "cuda")

        image = deprocess(image)
        # print(image.shape)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        image = preprocess(image, "cuda")

        image = octave + detail

        for j in range(iter_n):
            image = next_step_test(image, guided_features, model, device, target_layer_num, step_size, clip)

        detail = image - octave

    return deprocess(image)


if __name__ == "__main__":
    # va visim slojevima
    # kada radimo sa abgs slikom kao da levi gornji cosak uvek pojacava a donji desni nikad! Pitati aleksu

    # ABGSH
    n = 600
    img = 255.0 / 2.0 + 255.0 / 6 * np.random.randn(n, n, 3)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    noise_img = img

    # SLIKA
    filename = "Maskenbal2018"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = load_image(filepath + filename + extension)
    image = cv2.resize(image, (1000, 1000))

    filename = "noise"
    # image = noise_img

    plt.figure()
    plt.imshow(image)

    guide = load_image(filepath + "clock.jpg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    iter_n = 10
    # sloj 30 za ptice
    # sloj 16 krugovi
    target_layer_num = 29
    step_size = 0.04
    clip = True
    octave_n = 5
    octave_scale = 1.8

    image = deep_dream_test(image, guide, model, device, octave_n, octave_scale, iter_n, target_layer_num, step_size,
                            clip)

    plt.figure()
    plt.imshow(image)
    im = Image.fromarray(image)
    outpath = "data/fun_figures/"
    im.save(outpath + filename + "_iter_n_" + str(iter_n) + "_layer_" + str(target_layer_num) + "_step_" + str(
        step_size) + "_octave_n_" + str(octave_n) + "_octave_scale_" + str(octave_scale) + ".jpeg")
    plt.show()
