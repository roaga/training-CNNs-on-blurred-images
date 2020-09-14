import os
from PIL import Image
import shutil

root_dir = 'C:/Users/roaga/OneDrive/Documents/CS Projects/Summer_Internship_2019/hymenoptera_data_dark/hymenoptera_data_10'

def darken(img, mult):
    im1 = Image.open(img)
    width, height = im1.size
    im2 = Image.new("RGB", (width, height), "white")
    pixels = im2.load()        
    for i in range(width):
        for j in range(height):
            r, g, b = im1.getpixel((i, j))
            pixels[i, j] = (int(r * mult), int(g * mult), int(b * mult))
    return im2

def save(img, name, mult, endPath):
    if not os.path.exists(root_dir + "_" + str(mult) + "/" + endPath):
        os.makedirs(root_dir + "_" + str(mult) + "/" + endPath + "/", 0o777)
    img.save(root_dir + "_" + str(mult) + "/" + endPath + "/" + name, 'JPEG')

for mult in range(9): #different levels of darkening
    print("Multiplying RGB by " + str((mult + 1) * 0.1))
    for directory, subdirectories, files in os.walk(root_dir):
        for file in files:
            endPath = directory.split("\\")[-2] + "/" + directory.split("\\")[-1] #probably change to '/' for UNIX
            name = os.path.join(directory, file)
            newImage = darken(name, (mult + 1) * 0.1)
            save(newImage, file, mult + 1, endPath)