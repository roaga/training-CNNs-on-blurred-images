# trains on blurred images and classifies blurred images, looking at all combinations of train/test window sizes

# Banks 1978 paper:
# 1 month:  2.4 cyc/deg
# 2 month:  2.8 cyc/deg
# 3 month:  4 cyc/deg
# 224 pixels:
# 20 deg -> 11 pix in deg;  4.6 pix blur;  4 pix blur;  2.8 pix blur
# 4 deg -> 56 pix in deg; 23 pix blur (1 mo); 20 pix blur (2 mo); 14 pix blur (3 mo)

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

args = sys.argv 
modelType = args[1] # 'alexnet', 'squeezenet', 'vgg16'
numEpochs = args[2] # int

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
# data_dir = "C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagewoof-320_blur/"
# data_dir = "/home/cnslab/imagenet_blur/"
data_dir = "/u/erdos/cnslab/imagewoof-320_blur/"

classes = []
for directory, subdirectories, files in os.walk(data_dir + 'gaussian/imagewoof-320_1/val'):
    for file in files:
        if directory.split("\\")[-1] not in classes:
            classes.append(directory.split("\\")[-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train():
    for epoch in range(int(numEpochs)):  # loop over the dataset multiple times
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

            if i % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))

            if i % 10 == 9:    # check to break every 10 mini-batches
                if (prev_loss / 100 - running_loss / 100) < (0.1 * prev_loss / 100): # end if loss decreases by less than 10% # 0: # end if loss goes up 
                    return "Done"
                prev_loss = running_loss
                running_loss = 0.0

allAllAccs = []
for blurType in blurTypes: # multiple types of blur
    print(blurType)
    print('-' * 10)
    blurLevels = [1, 3, 5, 11, 23]
    allAccs = []
    for i in range(5): #5 levels of blur: 1, 3, 5, 11, 23
        mult = blurLevels[i]
        trainset = torchvision.datasets.ImageFolder(root=data_dir + blurType + '/imagewoof-320_' + str(mult) + '/train',
                                                transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                shuffle=True, num_workers=0)

        testset = torchvision.datasets.ImageFolder(root=data_dir + blurType + '/imagewoof-320_' + str(mult) + '/val',
                                                transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                shuffle=False, num_workers=0)

        if modelType == 'vgg16':
            net = torchvision.models.vgg16(pretrained=True)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(num_ftrs, len(classes))
        elif modelType == 'alexnet':
            net = torchvision.models.alexnet(pretrained=True)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(num_ftrs, len(classes))
        else:
            net = torchvision.models.squeezenet1_1(pretrained=True)
            net.classifier[1] = nn.Conv2d(512, len(classes), kernel_size=(1, 1), stride=(1, 1))
            net.num_classes = len(classes)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        net = net.to(device)

        train()
        # torch.save(net, 'C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/Models/TrainOnGaussianBlurModel_' + str(modelType) + '_' + str(mult) + '.pt')
        # torch.save(net, '/u/erdos/cnslab/CNNblur/models/TrainOnGaussianBlurModel_' + str(modelType) + '_' + str(mult) + '.pt')
        print('Finished Training on ' + blurType + ' with blur window of ' + str(mult))

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)

        accs = []
        for j in range(5):
            tempMult = blurLevels[j]
            correct = 0
            total = 0
            newTestSet = torchvision.datasets.ImageFolder(root=data_dir + blurType + '/imagewoof-320_' + str(tempMult) + '/val',
                                                    transform=transform)
            newTestLoader = torch.utils.data.DataLoader(newTestSet, batch_size=128,
                                                    shuffle=False, num_workers=0)
            with torch.no_grad():
                for data in newTestLoader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
            print('Accuracy: %f %%' % (acc))
            accs.append(acc)

        allAccs.append(accs)
    allAllAccs.append(allAccs)

print()
print(allAllAccs)

styles = ['r', 'b', 'y', 'k', 'g', 'c','m' ]
x = [1, 3, 5, 11, 23]

y = allAllAccs[0][0]
plt.plot(x, y, styles[0], label = 'Trained on Blur Size 1 px')
y = allAllAccs[0][1]
plt.plot(x, y, styles[1], label = 'Trained on Blur Size 3 px')
y = allAllAccs[0][2]
plt.plot(x, y, styles[2], label = 'Trained on Blur Size 5 px')
y = allAllAccs[0][3]
plt.plot(x, y, styles[3], label = 'Trained on Blur Size 11 px')
y = allAllAccs[0][4]
plt.plot(x, y, styles[4], label = 'Trained on Blur Size 23 px')

plt.legend()
plt.xlabel("Window Size (px)")
plt.ylabel("Percent Accuracy")
plt.title(str(modelType).capitalize() +  " Training and Classifying Blurred Images")
plt.ylim(0, 100)
plt.show()
plt.savefig('/u/erdos/cnslab/CNNblur/figures/' + str(modelType) +  'RealisticBlurCombos' + numEpochs + '_imagewoof.png')
# plt.savefig('/home/cnslab/CNNblur/figures/' + str(modelType) +  'AllRealisticBlurCombos_imagenet327Full.png')
# plt.savefig('C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/Figures/' + str(modelType) +  ' All Realistic Blur Combos_imagewoof.png')
