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
