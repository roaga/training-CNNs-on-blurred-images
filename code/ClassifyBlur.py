# only change paramaters of the last classification layer (fixed-feature extractor)
# written in conjunction with BlurImages.py

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
 
classNum = 7 # number of classes

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

accsDict = {}
def train_model(model, criterion, optimizer, scheduler, mult, num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    
    # add accuracy to dict
    accsDict['Dim: ' + str(mult)] = best_acc

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# model_conv = torchvision.models.alexnet(pretrained=True)
model_conv = torchvision.models.vgg16(pretrained=True)
# model_conv = torchvision.models.squeezenet1_1(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.classifier[6].in_features # ALEXNET, VGG16
model_conv.classifier[6] = nn.Linear(num_ftrs, 7) # ALEXNET, VGG16; change second parameter to num of classes

# model_conv.classifier[1] = nn.Conv2d(512, classNum, kernel_size=(1, 1), stride=(1, 1)) # SQUEEZENET1_1
# model_conv.num_classes = classNum # SQUEEZENET1_1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.classifier[6].parameters(), lr=0.001, momentum=0.9) # ALEXNET, VGG16
# optimizer_conv = optim.SGD(model_conv.classifier[1].parameters(), lr=0.001, momentum=0.9) # SQUEEZENET1_1

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

mult = 1
for i in range(7):
    # data_dir = 'C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320_blur/pixelate/imagenette-320_' + str(mult) # local
    data_dir = '/tmp/imagenette-320_blur/pixelate/imagenette-320_' + str(mult) # 150.108.68.131
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, mult, num_epochs=2)
    mult = (mult * 2) + 1

print(accsDict)