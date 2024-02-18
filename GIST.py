import numpy as np
import cv2 as cv
from scipy import fft


def _resize_and_crop(img: np.ndarray, new_shape: tuple[int, int]):
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


def preprocess(img: np.ndarray, fc=4):
    """Preprocess image through pre-filtering, whitening and local contrast normalisation
    :param fc: controls sigma in gaussian filter
    :param img: image to filter
    :return: filtered image
    """

    pad_w = 5

    # pad images to reduce boundary artifacts
    img = np.log(img + 1)
    img = np.pad(array=img, pad_width=((pad_w, pad_w), (pad_w, pad_w)), mode='symmetric')

    # dimensions of the padded image
    new_h, new_w = img.shape

    # make dimensions even through additional padding
    n = max(new_h, new_w)
    n += n % 2
    img = np.pad(array=img, pad_width=((0, n - new_h), (0, n - new_w)), mode='symmetric')

    # create gaussian filter
    fx, fy = np.meshgrid(np.arange(-n / 2, n / 2), np.arange(-n / 2, n / 2))
    sigma = fc / np.sqrt(np.log(2))
    gaussian_filter = fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (sigma ** 2)))

    # whitening
    output = img - np.real(fft.ifft2(fft.fft2(img) * gaussian_filter))

    # local contrast normalisation
    local_std = np.sqrt(np.abs(fft.ifft2(fft.fft2(output ** 2) * gaussian_filter)))
    output = output / (0.2 + local_std)

    # crop output to have the same size as the input
    output = output[pad_w:new_h - pad_w, pad_w:new_w - pad_w]

    return output
