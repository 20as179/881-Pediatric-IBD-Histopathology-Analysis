#converting H&E .jpg to multichannel
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb

def Convert2Multichannel(input_path, output_path):
    # LOAD IMAGES
    images = cv2.imread(input_path)

    # CONVERT TO H&E AND SEPARATE CHANNELS
    image_rgb = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)  # convert to RGB
    image_hed = rgb2hed(image_rgb)  # convert RGB to HED

    null = np.zeros_like(image_hed[:, :,
                         0])  # https://scikit-image.org/docs/0.25.x/auto_examples/color_exposure/plot_ihc_color_separation.html
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ihc_h = hed2rgb(np.stack((image_hed[:, :, 0], null, null), axis=-1))
    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")
    print("test2")

    ihc_e = hed2rgb(np.stack((null, image_hed[:, :, 1], null), axis=-1))
    ax[2].imshow(ihc_e)
    ax[2].set_title("Eosin")
    print("test")

    ihc_d = hed2rgb(np.stack((null, null, image_hed[:, :, 2]), axis=-1))
    ax[3].imshow(ihc_d)
    ax[3].set_title("DAB")
    print("test3")

    # normalize H&E channels to [0, 1] range
    hematoxylin = (image_hed[:, :, 0] - image_hed[:, :, 0].min()) / (
            image_hed[:, :, 0].max() - image_hed[:, :, 0].min())
    eosin = (image_hed[:, :, 1] - image_hed[:, :, 1].min()) / (image_hed[:, :, 1].max() - image_hed[:, :, 1].min())

    # create an empty channel
    empty_channel = np.zeros_like(hematoxylin)
    # combine channels
    multichannel_image = np.stack((hematoxylin, eosin, empty_channel), axis=-1)
    print(f"Multichannel image saved as {output_path}")
    return multichannel_image

    # save the final multichannel image
    imwrite(output_path, multichannel_image)

def main():
    input_path =
    output_path =

    multichannel_image = Convert2Multichannel(input_path, output_path)
    plt.imshow(multichannel_image, cmap = 'Blues')
    plt.title("Original??")
    #print('hi')
    #plt.imshow(multichannel_image, cmap = 'Greens')


if __name__ == '__main__':
    main()