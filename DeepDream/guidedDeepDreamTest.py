from utils import *


def next_step_guided(img, guide, model, device, target_layer_num=1, step_size=0.02, clip=True):
    # img_tensor : current tensor to ascent
    # model : chosen model
    # target_layer_num : order of layer in model (1..end)
    # step_size : learning rate
    # clip : saturation of image pixels, makes sure that in the end they are 0..255
    # init with current image

    guide_tensor = torch.tensor(guide, requires_grad=False).to(device)
    guide_features = propagate(guide_tensor, model, target_layer_num)

    img_tensor = torch.tensor(img, requires_grad=True).to(device)
    img_tensor.retain_grad()

    # forward img to target layer
    target_layer_output = propagate(img_tensor, model, target_layer_num)

    model.zero_grad()
    # target_layer_output.norm().backward()
    print(target_layer_output.shape)
    print(guide_features.shape)
    print(torch.dot(target_layer_output, guide_features))
    torch.dot(target_layer_output, guide_features).backward()

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


if __name__ == "__main__":
    filename = "clock"
    extension = ".jpg"
    filepath = "data/input_images/"
    guide = loadImage(filepath + filename + extension)
    guide = cv2.resize(guide, (200, 200))
    guide = prepareImage(guide)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loadModel(device)
    target_layer_num = 16

    guide_tensor = torch.tensor(guide, requires_grad=False).to(device)
    guide_features = propagate(guide_tensor, model, target_layer_num)
    guide_features = guide_features[0]

    filename = "Maskenbal2018"
    extension = ".jpg"
    filepath = "data/input_images/"
    image = loadImage(filepath + filename + extension)
    # image = cv2.resize(image, (150, 150))
    image = prepareImage(image)

    img_tensor = torch.tensor(image, requires_grad=True).to(device)
    img_tensor.retain_grad()
    target_layer_output = propagate(img_tensor, model, target_layer_num)

    x = target_layer_output[0].data.cpu().numpy()
    y = guide_features.data.cpu().numpy()
    ch = x.shape[0]
    x = x.reshape(ch, -1)
    y = y.reshape(ch, -1)
    A = x.T.dot(y)
    z = y[:, A.argmax(1)]
    print(target_layer_output)


    # iter_n = 10
    # step_size = 0.04
    # clip = True
    # octave_n = 5
    # octave_scale = 1.8

    # image = next_step_guided(image, guide, model, device, target_layer_num, step_size, clip=True)

    # image = deprocess(image)

    # plt.figure()
    # plt.imshow(image)
    # plt.show()
