import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy
from PIL import Image

def getMatrix(image):
    data = list(image.getdata())
    width, height = image.size
    matrix = np.array(data).reshape(height,width)
    return matrix

def rescale(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix * 255 / matrix.max()
    return matrix

def getData(matrix):
    data = list(matrix.reshape(matrix.shape[0]*matrix.shape[1]))
    return data

def blur(matrix, a, b, T):
    m, n = matrix.shape
    u, v = np.ogrid[-m/2:m/2, -n/2:n/2]
    x = u * a + v * b
    result = T * np.sinc(x) * np.exp(-1j*np.pi*x)
    return result


im1 = cv2.imread('C:/Users/roaga/OneDrive/Pictures/dog.jpg')
b,g,r = cv2.split(im1) 


fourierMatb = np.fft.fft2(b)
blurredFMatb = fourierMatb * blur(fourierMatb, a=0.1, b=0.1, T=1)
blurredMatb = np.fft.ifft2(blurredFMatb)
print(blurredMatb)
fourierMatg = np.fft.fft2(g)
blurredFMatg = fourierMatg * blur(fourierMatg, a=0.1, b=0.1, T=1)
blurredMatg = np.fft.ifft2(blurredFMatg)
print(blurredMatg)
fourierMatr = np.fft.fft2(r)
blurredFMatr = fourierMatr * blur(fourierMatr, a=0.1, b=0.1, T=1)
blurredMatr = np.fft.ifft2(blurredFMatr)
print(blurredMatr)


# plt.subplot(121),plt.imshow(blurredMatb)
# plt.show()