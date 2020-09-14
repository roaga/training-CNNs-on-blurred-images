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

transform = transforms.Compose([         
 transforms.Resize(256),                 
 transforms.CenterCrop(224),     
 transforms.ToTensor(),     
 transforms.Normalize(             
 mean=[0.485, 0.456, 0.406],          
 std=[0.229, 0.224, 0.225]           
 )])

data_dir = "C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagenette-320"
# data_dir = "/tmp/imagenette-320"
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
trainset = torchvision.datasets.ImageFolder(root=data_dir + '/train',
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
testset = torchvision.datasets.ImageFolder(root=data_dir + '/val',
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = []
for directory, subdirectories, files in os.walk(data_dir + '/val'):
    for file in files:
        if directory.split("\\")[-1] not in classes:
            classes.append(directory.split("\\")[-1])

# functions to show an image

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = models.alexnet(pretrained = True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

def train():
    print("Beginning Training")
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

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                if (prev_loss / 100 - running_loss / 100) < (0.001 * prev_loss / 100): # end if loss decreases by less than .01%
                    return "Done"
                prev_loss = running_loss
                running_loss = 0.0

train()
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))
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

print('Accuracy: %f %%' % (
    100 * correct / total))

# print('Parameters: ' + str(list(net.parameters())))
# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# # Print optimizer's state_dict
# # print("Optimizer's state_dict:")
# # for var_name in optimizer.state_dict():
# #     print(var_name, "\t", optimizer.state_dict()[var_name])