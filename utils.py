import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def plot_rgb(img):
    plt.figure(figsize=(9, 6))
    return plt.imshow(img)


def plot_bgr(img):
    plt.figure(figsize=(9, 6))
    return plt.imshow(to_rgb(img))


def plot_gray(img):
    plt.figure(figsize=(9, 6))
    return plt.imshow(img, cmap="Greys_r")
