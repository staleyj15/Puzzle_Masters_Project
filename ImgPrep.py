from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import re


def file_names(directory):

    """
    store all file names in list for processing
    """

    paths = os.listdir("puzzle_scans/puzzle_1")
    return [
        directory + "/" + file for file in paths if re.search(".jpg", file) is not None
    ]


def read_img(img_path):

    """
    read in image and store as array
    """

    img = Image.open(img_path)
    img_array = np.array(img, dtype="int32")
    img.close()
    return img_array


def reshape_array(arr, starting_dim):

    """
    flatten array to (shape # of pixels, 3)
    """
    if starting_dim == 3:
        return np.reshape(arr, (arr.shape[0] * arr.shape[1], arr.shape[2]), order="C")
    else:
        return np.reshape(arr, (1, arr.shape[0] * arr.shape[1]), order = "C")


def split_img(arr, nrow, ncol):

    """
    split image into nrow by ncol smaller images
    """

    def find_splits(shape, num):
        split = math.floor(shape / num)
        return [split * i for i in range(1, num)]

    def equal_pixels(list_arr, ax):
        extra = list_arr[-1].shape[ax] - list_arr[0].shape[ax] + 1
        rem_indx = [list_arr[-1].shape[ax] - n for n in reversed(range(1, extra))]
        replace = np.delete(list_arr[-1], rem_indx, axis=ax)
        list_arr.pop(-1)
        return list_arr + [replace]

    pieces = []
    rows = equal_pixels(np.split(arr, find_splits(arr.shape[0], nrow)), 0)

    for row in rows:
        cols = equal_pixels(np.split(row, find_splits(row.shape[1], ncol), axis=1), 1)
        pieces += cols

    return pieces


def remove_background(arr):
    
    """
    remove background from image
    """
    
    img = ski.color.rgb2gray(arr)
    sobel = filters.sobel(img)
    blurred = filters.gaussian(sobel, sigma=10)

    bgremoved = arr.copy()
    for i in range(bgremoved.shape[0]):
        for j in range(bgremoved.shape[1]):
            if blurred[i,j] <= 0.02:
                bgremoved[i,j] = [0, 0, 0]
                
    return bgremoved

def glcm(arr):
    
    """
    create gray level co-occurence matrix for image with background removed and calculate texture features
    """
    
    img = ski.color.rgb2gray(arr)
    img = (img * 255).astype(int)
    P = skft.greycomatrix(img, [5], [np.pi/8, (3*np.pi)/8, (5*np.pi)/8, (7*np.pi)/8], levels=256)
    P2 = np.empty([P.shape[0]-1,P.shape[1]-1,P.shape[2],P.shape[3]], dtype = int)
    P2[:,:,0,0] = P[1:,1:,0,0]
    P2[:,:,0,1] = P[1:,1:,0,1]
    P2[:,:,0,2] = P[1:,1:,0,2]
    P2[:,:,0,3] = P[1:,1:,0,3]
    
    features = np.array([])
    features = np.append(features, skft.greycoprops(P2, 'contrast').flatten())
    features = np.append(features, skft.greycoprops(P2, 'dissimilarity').flatten())
    features = np.append(features, skft.greycoprops(P2, 'homogeneity').flatten())
    features = np.append(features, skft.greycoprops(P2, 'ASM').flatten())
    features = np.append(features, skft.greycoprops(P2, 'energy').flatten())
    features = np.append(features, skft.greycoprops(P2, 'correlation').flatten())
    
    return features
