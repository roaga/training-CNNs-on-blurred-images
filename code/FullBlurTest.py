# trains on blurred images and classifies blurred images, trains on unblurred images and classifies blurred images, for 3 models

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

transform = transforms.Compose([         
 transforms.Resize(256),                 
 transforms.CenterCrop(224),     
 transforms.ToTensor(),     
 transforms.Normalize(             
 mean=[0.485, 0.456, 0.406],          
 std=[0.229, 0.224, 0.225]           
 )])

# blurTypes = ['average', 'gaussian', 'pixelate']
blurTypes = ['gaussian']
# data_dir = "C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320_blur/"
data_dir = "/u/erdos/cnslab/CNNblur/imagenette-320_blur/"

classes = []
for directory, subdirectories, files in os.walk(data_dir + 'average/imagenette-320_1/val'):
    for file in files:
        if directory.split("\\")[-1] not in classes:
            classes.append(directory.split("\\")[-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train():
    for epoch in range(250):  # loop over the dataset multiple times
        prev_loss = 100000.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99: # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))

            if i % 10 == 9:    # check to break every 10 mini-batches
                if (prev_loss / 100 - running_loss / 100) < (0.1 * prev_loss / 100): # end if loss decreases by less than 10%
                    return "Done"
                prev_loss = running_loss
                running_loss = 0.0

models = ['AlexNet', 'SqueezeNet', 'VGG16'] #TODO add 'VGG16'
allAllAllAccs = [] # [AlexNet (Unblurred [BlurTypes], Blurred [BlurTypes]), Squeezenet (Unblurred [BlurTypes], Blurred [BlurTypes])]
for modelType in models:
    criterion = nn.CrossEntropyLoss()
    allAllAccs = []
    for run in range(2):
        if run == 0: # train on unblurred
            trainset = torchvision.datasets.ImageFolder(root=data_dir + 'gaussian/imagenette-320_1/train',
                                        transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                        shuffle=True, num_workers=0)

            if modelType == 'VGG16':
                net = torchvision.models.vgg16(pretrained=True)
                num_ftrs = net.classifier[6].in_features
                net.classifier[6] = nn.Linear(num_ftrs, len(classes))
            elif modelType == 'AlexNet':
                net = torchvision.models.alexnet(pretrained=True)
                num_ftrs = net.classifier[6].in_features
                net.classifier[6] = nn.Linear(num_ftrs, len(classes))
            else:
                net = torchvision.models.squeezenet1_1(pretrained=True)
                net.classifier[1] = nn.Conv2d(512, len(classes), kernel_size=(1, 1), stride=(1, 1))
                net.num_classes = len(classes)
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = net.to(device)
            
            train()
            # torch.save(net, 'C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/Models/TrainOnUnblurModel_' + str(modelType) + '.pt')
            torch.save(net, '/u/erdos/cnslab/CNNblur/models/TrainOnUnblurModel_' + str(modelType) + '.pt')
            print('Finished Training on Unblurred Images')

            allAccs = []
            for blurType in blurTypes: # multiple types of blur
                print(blurType)
                print('-' * 10)
                mult = 1
                accs = {}
                for i in range(7): #7 levels of blur: 1, 3, 7, 15, 31, 63, 127
                    testset = torchvision.datasets.ImageFolder(root=data_dir + 'gaussian/imagenette-320_' + str(mult) + '/val',
                                                    transform=transform)
                    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                    shuffle=False, num_workers=0)
                    dataiter = iter(testloader)
                    images, labels = dataiter.next()
                    images = images.to(device)
                    labels = labels.to(device)

                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    acc = 100 * correct / total
                    print('Blur window of ' + str(mult))
                    print('Accuracy: %f %%' % (acc))
                    accs['Dim ' + str(mult)] = acc
                    mult = mult * 2 + 1
                allAccs.append(accs)
            allAllAccs.append(allAccs)
        else: # train on blurred
            allAccs = []
            for blurType in blurTypes: # multiple types of blur
                print(blurType)
                print('-' * 10)
                mult = 1
                accs = {}
                for i in range(7): #7 levels of blur: 1, 3, 7, 15, 31, 63, 127
                    trainset = torchvision.datasets.ImageFolder(root=data_dir + blurType + '/imagenette-320_' + str(mult) + '/train',
                                                            transform=transform)
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                            shuffle=True, num_workers=0)

                    testset = torchvision.datasets.ImageFolder(root=data_dir + blurType + '/imagenette-320_' + str(mult) + '/val',
                                                            transform=transform)
                    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                            shuffle=False, num_workers=0)

                    if modelType == 'VGG16':
                        net = torchvision.models.vgg16(pretrained=True)
                        num_ftrs = net.classifier[6].in_features
                        net.classifier[6] = nn.Linear(num_ftrs, len(classes))
                    elif modelType == 'AlexNet':
                        net = torchvision.models.alexnet(pretrained=True)
                        num_ftrs = net.classifier[6].in_features
                        net.classifier[6] = nn.Linear(num_ftrs, len(classes))
                    else:
                        net = torchvision.models.squeezenet1_1(pretrained=True)
                        net.classifier[1] = nn.Conv2d(512, len(classes), kernel_size=(1, 1), stride=(1, 1))
                        net.num_classes = len(classes)
                    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    net = net.to(device)
                    
                    train()
                    # torch.save(net, 'C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/Models/TrainOn' + blurType + 'BlurModel_' + str(modelType) + '_' + str(mult) + '.pt')
                    torch.save(net, '/u/erdos/cnslab/CNNblur/models/TrainOn' + blurType + 'BlurModel_' + str(modelType) + '_' + str(mult) + '.pt')
                    print('Finished Training on ' + blurType + ' with blur window of ' + str(mult))

                    dataiter = iter(testloader)
                    images, labels = dataiter.next()
                    images = images.to(device)
                    labels = labels.to(device)

                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    acc = 100 * correct / total
                    print('Accuracy: %f %%' % (acc))
                    accs['Dim ' + str(mult)] = acc
                    mult = mult * 2 + 1
                allAccs.append(accs)
            allAllAccs.append(allAccs)
    allAllAllAccs.append(allAllAccs)

styles = ['r', 'b', 'r--', 'b--', 'r:', 'b:']
x = [1, 3, 7, 15, 31, 63, 127]

print(allAllAllAccs)
y = list(allAllAllAccs[0][0][0].values())
plt.plot(x, y, styles[0], label = 'AlexNet Trained on Unblurred')
y = list(allAllAllAccs[0][1][0].values())
plt.plot(x, y, styles[1], label = 'AlexNet Trained on Blurred')
y = list(allAllAllAccs[1][0][0].values())
plt.plot(x, y, styles[2], label = 'SqueezeNet Trained on Unblurred')
y = list(allAllAllAccs[1][1][0].values())
plt.plot(x, y, styles[3], label = 'SqueezeNet Trained on Blurred')
y = list(allAllAllAccs[2][0][0].values())
plt.plot(x, y, styles[4], label = 'VGG16 Trained on Unblurred')
y = list(allAllAllAccs[2][1][0].values())
plt.plot(x, y, styles[5], label = 'VGG16 Trained on Blurred')

plt.legend()
plt.xlabel("Window Size (px)")
plt.ylabel("Percent Accuracy")
plt.title("CNN Performance Classifying Blurred Images")
plt.ylim(0, 100)
plt.show()
plt.savefig('/u/erdos/cnslab/CNNblur/figures/Blur and Unblur Performance.png')