import os
import shutil
import numpy as np
import cv2
import math

blurType = "gaussian" # "average", "gaussian", "pixelate"
# root_dir = 'C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagewoof-320'
# root_dir = '/u/erdos/cnslab/imagenet'
root_dir = '/home/cnslab/imagenet'

def blurImage(img, dim, blurType):
    im1 = cv2.imread(img)
    width, height, _ = im1.shape
    centerx = math.floor(width / 2)
    centery = math.floor(height / 2)
    im1 = im1[centerx - 187 : centerx + 188, centery - 187 : centery + 188] # center crop of 375

    if blurType == "average": 
        im2 = cv2.blur(im1, (dim, dim))
    elif blurType == "gaussian":
        im2 = cv2.GaussianBlur(im1,(dim, dim), 0)
    elif blurType == "pixelate":
        w, h = (int(width / dim), int(height / dim))
        temp = cv2.resize(im1, (w, h), interpolation=cv2.INTER_LINEAR)
        im2 = cv2.resize(temp, (height, width), interpolation=cv2.INTER_NEAREST)
    else:
        print("INVALID FILTER")
        return im1 
    return im2

def save(img, name, mult, endPath, trainOrVal):
    if not os.path.exists("/home/cnslab/imagenet_blur/gaussian/imagenet_" + str(mult) + "/" + trainOrVal + "/" + endPath):
        os.makedirs("/home/cnslab/imagenet_blur/gaussian/imagenet_" + str(mult) + "/" + trainOrVal + "/" + endPath + "/", 0o777)
    cv2.imwrite("/home/cnslab/imagenet_blur/gaussian/imagenet_" + str(mult) + "/" + trainOrVal + "/" + endPath + "/" + name, img)

blurLevels = [1, 3, 5, 11, 23]
omitted = 0
corrupted = 0
for i in range(len(blurLevels)): # different levels of blurring
    mult = blurLevels[i]
    print("Working with square kernel dimension = " + str(mult))
    for directory, subdirectories, files in os.walk(root_dir):
        num = 1
        for file in files:
            if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                if num == 1:
                    trainOrVal = 'train'
                else:
                    trainOrVal = 'val'
                try:
                    # endPath = directory.split("\\")[-2] + "/" + directory.split("\\")[-1] #probably change to '/' for UNIX, '\\' for Windows (except 2nd slash, which is '/')
                    endPath = directory.split("/")[-1] #probably change to '/' for UNIX, '\\' for Windows (except 2nd slash, which is '/')
                    name = os.path.join(directory, file)
                    x, y, z = cv2.imread(name).shape
                    if x > 375 and y > 375:
                        newImage = blurImage(name, mult, blurType)
                        save(newImage, file, mult, endPath, trainOrVal)
                        num *= -1
                    else:
                        omitted += 1
                except:
                    corrupted += 1
print(str(omitted) + " pictures less than 375 x 375 px were omitted")
print(str(corrupted) + " pictures were omitted because of corrupt data")