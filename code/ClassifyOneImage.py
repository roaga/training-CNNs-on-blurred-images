import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
 mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225]
 )])

img = Image.open('C:/Users/roaga/OneDrive/Pictures/elephant.jpg')
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

net = models.alexnet(pretrained=True)
net.eval()

out = net(batch_t)

with open('C:/Users/roaga/OneDrive/Documents/CS Projects/Summer_Internship_2019/imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

#print top 1
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(classes[index[0]], percentage[index[0]].item())

#print top 5
_, indices = torch.sort(out, descending=True)
print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])