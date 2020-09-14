import cv2
import numpy as np
import os
import math
import matplotlib
import matplotlib.pyplot as plt

# data_dir = "C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320_blur/gaussian/imagenette-320_1/"
# data_dir = "/home/cnslab/imagenette-320_blur/gaussian/imagenette-320_1/"
data_dir = "/u/erdos/cnslab/imagenette-320_blur/gaussian/imagenette-320_1/train/"

imageCount = 0
imageList = []

for directory, subdirectories, files in os.walk(data_dir):
    num = 0
    for file in files:
        if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            name = os.path.join(directory, file)
            x, y, z = cv2.imread(name).shape
            if x > 375 and y > 375 and num < 10:
                imageCount += 1
                imageList.append(os.path.join(directory, file))
                num += 1
print(imageList)

matrix = np.zeros((imageCount, imageCount))

def getDistance(im1, im2):
    im1 = im1.astype(float)
    return math.sqrt(np.sum((im1[:] - im2[:]) ** 2))

count = 0
for pos1 in range(len(imageList)):
    im1 = cv2.imread(imageList[pos1])
    im1 = im1[0:224:3,0:224:3,:].mean(2) 
    im1 = cv2.resize(im1, (125, 125))
    for pos2 in range(len(imageList)):
        count += 1
        im2 = cv2.imread(imageList[pos2])
        im2 = im2[0:224:3,0:224:3,:].mean(2) 
        im2 = cv2.resize(im2, (125, 125))
        distance = getDistance(im1, im2)
        print(str(count) + " / " + str(imageCount ** 2))
        matrix[pos1, pos2] = distance

print(np.mean(matrix))

# fig, ax = plt.subplots()
# ax.xaxis.tick_top()
# plt.imshow(np.asarray(matrix))
# plt.colorbar()
# # plt.show()

# plt.savefig('/home/cnslab/CNNblur/figures/ConfusionMatrix_imagenet.png')