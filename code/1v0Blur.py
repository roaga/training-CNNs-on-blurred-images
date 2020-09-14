import cv2
import math
import numpy as np

def getDistance(im1, im2):
    im1 = im1.astype(float)
    return math.sqrt(np.sum((im1[:] - im2[:]) ** 2))

im1 = cv2.imread('C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320/train/n01440764/n01440764_18.JPEG')
im1 = cv2.resize(im1, (224, 224))
im1 = cv2.GaussianBlur(im1, (1, 1), 0)
im2 = cv2.imread('C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320/train/n01440764/n01440764_18.JPEG')
im2 = cv2.resize(im2, (224, 224))
print(getDistance(im1, im2))

# CONCLUSION: NO BLUR AND BLURRING WITH A WINDOW OF 1 ARE IDENTICAL WITH A EUCLIDEAN DISTANCE OF 0.0