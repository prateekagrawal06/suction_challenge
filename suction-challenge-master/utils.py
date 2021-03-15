import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from PIL import Image


def rgb2array(data_path,
              desired_size=None,
              expand=False,
              hwc=True,
              show=False):
    """Loads a 24-bit PNG RGB image as a 3D or 4D numpy array."""
    img = Image.open(data_path).convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.float32)
    if show:
        plt.imshow(x.astype(np.uint8), interpolation='nearest')
    if not hwc:
        x = np.transpose(x, [2, 0, 1])
    if expand:
        x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    return x


def depth2array(data_path,
                desired_size=None,
                expand=False,
                hwc=True,
                depth_scale=1e-3,
                show=False):
    """Loads a 16-bit PNG depth image as a 3D or 4D numpy array."""
    img = Image.open(data_path)
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.float32)
    if show:
        plt.imshow(x, cmap='gray', interpolation='nearest')
    x = np.expand_dims(x, axis=0)
    if hwc:
        x = np.transpose(x, [1, 2, 0])
    if expand:
        x = np.expand_dims(x, axis=0)
    # Intel RealSense depth cameras store
    # depth in millimers (1e-3 m) so we convert
    # back to meters
    x *= depth_scale
    x = x.astype(np.float32)
    return x


def label2array(data_path, desired_size=None, hwc=True, show=False):
    """Loads an 8-bit grayscale PNG image as a 3D numpy array."""
    img = Image.open(data_path)
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = np.array(img, dtype=np.int64)[..., 1]
    if show:
        plt.imshow(x, norm=MidpointNorm(0, 255, 1), interpolation='nearest')
    x = np.expand_dims(x, axis=0)
    if hwc:
        x = np.transpose(x, [1, 2, 0])
    x[x == 255] = 1.
    return x
