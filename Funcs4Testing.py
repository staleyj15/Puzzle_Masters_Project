import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def display_img(array):

    """
    display image
    """

    arr = array.astype(dtype="uint8")
    img = Image.fromarray(arr, "RGB")
    plt.figure()
    plt.imshow(np.asarray(img))
    

def display_cluster_imgs(pictures, labels, cluster_k):
    
    '''
    display all images in cluster k
    '''
    
    print(f'\nImages in cluster {cluster_k}:\n')
    displayIndx = np.where(labels == cluster_k)[0]
    for indx in displayIndx:
        display_img(pictures[indx])
        

def img_scatter(Z, images, k, zoom=0.5):
    
    '''
    plot node images on top of scatter plot
    '''
    
    m = len(images)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(
        "2D Components from Isomap of Puzzle Piece Images", fontsize=16, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xsize = (max(Z[:, 0]) - min(Z[:, 0])) * 0.04
    ysize = (max(Z[:, 1]) - min(Z[:, 1])) * 0.04

    # show 30 images on the plot
    for i in range(k):
        img_num = np.random.randint(0, m)
        x0 = Z[img_num, 0] - (xsize / 2)
        y0 = Z[img_num, 1] - (ysize / 2)
        x1 = Z[img_num, 0] + (xsize / 2)
        y1 = Z[img_num, 1] + (ysize / 2)

        arr = images[img_num].astype(dtype="uint8")
        img = Image.fromarray(arr, "RGB")

        ax.imshow(
            img,
            aspect="auto",
            zorder=100000,
            extent=(x0, x1, y0, y1),
        )

    ax.scatter(Z[:, 0], Z[:, 1])
    plt.show()



