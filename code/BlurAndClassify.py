import os
import shutil
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import time
import copy
import sys

# python BlurAndClassify.py <modelType>
args = sys.argv 
modelType = args[1] # 'alexnet', 'squeezenet', 'vgg16'

classNum = 7
blurTypeList = ['average', 'gaussian', 'pixelate']
data_dir = 'C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320' # local
# data_dir = '/tmp/imagenette-320' # linux server

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

transformNew = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

criterion = nn.CrossEntropyLoss()
if modelType == 'vgg16':
    model_conv = torchvision.models.vgg16(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.classifier[6].in_features
    model_conv.classifier[6] = nn.Linear(num_ftrs, classNum)
    optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=0.001, momentum=0.9)
elif modelType == 'alexnet':
    model_conv = torchvision.models.alexnet(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.classifier[6].in_features
    model_conv.classifier[6] = nn.Linear(num_ftrs, classNum)
    optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=0.001, momentum=0.9)
else:
    model_conv = torchvision.models.squeezenet1_1(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    model_conv.classifier[1] = nn.Conv2d(512, classNum, kernel_size=(1, 1), stride=(1, 1))
    model_conv.num_classes = classNum
    optimizer_conv = optim.SGD(model_conv.classifier[1].parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = model_conv.to(device)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

def blurImage(img, dim, blurType):
    im1 = cv2.imread(img)
    # transforms.Resize(256)
    im1 = cv2.resize(im1, (256, 256))
    # transforms.CenterCrop(224)
    im1 = im1[16:240, 16:240]
    
    if blurType == "average": 
        im2 = cv2.blur(im1, (dim, dim))
    elif blurType == "gaussian":
        im2 = cv2.GaussianBlur(im1,(dim, dim), 0)
    elif blurType == "pixelate":
        width, height, _ = im1.shape
        w, h = (int(width / dim), int(height / dim))
        temp = cv2.resize(im1, (w, h), interpolation=cv2.INTER_LINEAR)
        im2 = cv2.resize(temp, (height, width), interpolation=cv2.INTER_NEAREST)
    return im2

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# fix prediction layer weights in model so it can predict among the right number of classes
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=0)
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['val'].classes
inputs, classes = next(iter(dataloaders['train']))
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=1)

allAccs = []
for x in range(len(blurTypeList)):
    mult = 1
    accs = {}
    blurType = blurTypeList[x]
    print("Blur Type : " + str(blurType))
    for i in range(7): #7 levels of processing: 1, 3, 7, 15, 31, 63, 127 size windows
        corrects = 0
        print("Working with square kernel dimension = " + str(mult))
        for directory, subdirectories, files in os.walk(os.path.join(data_dir, 'val')):
            for file in files:
                name = os.path.join(directory, file)
                newImage = blurImage(name, mult, blurType)
                newImage = transformNew(newImage)
                newImage = torch.unsqueeze((newImage).float(), 0)
                out = model_conv(newImage)
                _, index = torch.max(out, 1)
                if str(class_names[index[0]]) == directory.split("\\")[-1]:
                    corrects += 1

        acc = corrects / dataset_sizes['val'] * 100
        accs['Dim ' + str(mult)] = acc
        mult = (mult * 2) + 1
        print(acc)
        
    print(accs)
    allAccs.append(accs)
    print('-' * 10)

# plot all accuracies over all types of blur
styles = ['r', 'b', 'g', 'y', 'k']
x = [1, 3, 7, 15, 31, 63, 127]
for num in range(len(blurTypeList)):
    y = list(allAccs[num].values())
    plt.plot(x, y, styles[num], label = blurTypeList[num])

plt.legend()
plt.xlabel("Window Size (px)")
plt.ylabel("Percent Accuracy")
plt.title(str(modelType) + " classifying blurred images")
plt.show()