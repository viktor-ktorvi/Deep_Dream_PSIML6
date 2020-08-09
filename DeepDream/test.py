from utils import *


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

    target_layer_num = 11
    features = propagate(img_tensor, model, target_layer_num)

    gram = gram_matrix(features)
    print(gram.shape)


    image = deprocess(img_tensor)
    plt.figure()
    plt.imshow(image)

    plt.show()
