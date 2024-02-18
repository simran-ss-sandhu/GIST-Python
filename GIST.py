import numpy as np
import cv2 as cv


def resize_and_crop(img: np.ndarray, new_shape: tuple[int, int]):
    """Resize image to a new shape whilst maintaining the original aspect ratio
    :param img: the image to resize
    :param new_shape: the desired shape
    :return: image with the desired shape
    """

    # resize image while keeping aspect ratio
    scaling = max(new_shape[0] / img.shape[0], new_shape[1] / img.shape[1])
    resize_shape = tuple(np.round(np.array([img.shape[1], img.shape[0]]) * scaling).astype(int))
    img = cv.resize(src=img, dsize=resize_shape, interpolation=cv.INTER_LINEAR)

    # crop image to get desired shape
    starting_row = (img.shape[0] - new_shape[0]) // 2
    starting_col = (img.shape[1] - new_shape[1]) // 2
    img = img[starting_row:starting_row + new_shape[0], starting_col:starting_col + new_shape[1]]

    return img
