import cv2
import numpy as np
import os
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.spatial.distance

data_dir = "/u/erdos/cnslab/imagewoof-320_blur/gaussian/imagewoof-320_1/train/"
# data_dir = "/home/cnslab/imagenet/"
# data_dir = "C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagewoof-320/train/"

def getDistance(im1, im2):
    im1 = im1.astype(float)
    # return math.sqrt(np.sum((im1[:] - im2[:]) ** 2))
    return scipy.spatial.distance.cdist(im1, im2, 'euclidean')

os.chdir(data_dir)
dir = os.listdir()
catCount = 0
cats = []
for i in dir:
    print(i)
    if i[-1].isdigit() and i[0] == 'n':
    # if i[-1] == '_' and i[0] == 'n':
        catCount += 1
        cats.append(i)
print(str(catCount) + " classes found")
matrix = np.zeros((catCount, catCount))
valList = []
diagonalVals = []

for i in range(len(cats)):
    os.chdir(data_dir + cats[i])
    images1 = os.listdir()
    images1 = images1[0:10]
    for j in range(i, len(cats)):
        dists = []
        print("Categories: " + cats[i] + ', ' + cats[j])
        images2 = []
        os.chdir(data_dir + cats[j])
        images2 = os.listdir()
        images2 = images2[0:10]
        for image1 in images1:
            for image2 in images2:
                if image1.endswith('.JPEG') and image2.endswith('.JPEG'):
                    im1 = cv2.imread(data_dir + cats[i] + '/' + image1)
                    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                    im1 = cv2.resize(im1, (75, 75))

                    im2 = cv2.imread(data_dir + cats[j] + '/' + image2)
                    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
                    im2 = cv2.resize(im2, (75, 75))
                    dists.append(getDistance(im1, im2))

        avgDist = np.mean(dists)
        matrix[i, j] = avgDist
        matrix[j, i] = avgDist
        valList.append(avgDist)
        if i == j: 
            diagonalVals.append(avgDist)
    
diagonalMins = sorted(diagonalVals)[0 : 10]
diagonalMaxes = sorted(diagonalVals, reverse=True)[0:10]

print("Lowest values on diagonal/most internally similar: " + str(diagonalMins))
print("Highest values on diagonal/most internally different: " + str(diagonalMaxes))

# create spreadsheet
# np.savetxt("/home/cnslab/CNNblur/figures/ConfusionMatrixData_imagenet327.csv", matrix, delimiter=",")

print(np.mean(matrix))
print(np.std(matrix))
# matrix visualisation
# fig, ax = plt.subplots()
# # ax.xaxis.tick_top()
# # plt.imshow(np.asarray(matrix))
# # plt.colorbar()
# cmap = mpl.cm.get_cmap('viridis', 256)
# psm = ax.pcolormesh(matrix, cmap=cmap, rasterized=True, vmin=300, vmax=1300)
# fig.colorbar(psm, ax=ax)
# plt.show()
# # plt.savefig('/u/erdos/cnslab/CNNblur/figures/ConfusionMatrix_imagenet.png')
# plt.savefig('/home/cnslab/CNNblur/figures/ConfusionMatrixFixed_imagenet327.png')

# # histogram visualization
# fig, ax = plt.subplots()
# hist, bins = np.histogram(valList, bins='auto')
# center = (bins[:-1] + bins[1:]) / 2
# width = np.diff(bins)
# ax.bar(center, hist, align='center', width=width)
# plt.xlim(300, 1300)
# plt.title("Imagewoof Confusion Matrix Histogram")
# plt.show()
# # plt.savefig("/u/erdos/cnslab/CNNblur/figures/ConfusionMatrixHistogram_imagenet90.png")
# plt.savefig('/home/cnslab/CNNblur/figures/ConfusionMatrixHistogramFixed_imagenet327.png')


