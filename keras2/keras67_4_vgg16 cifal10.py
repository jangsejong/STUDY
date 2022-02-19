import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(123)
if device =='cuda':
    torch.cuda.manual_seed_all(123)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt 
import numpy as np 
# %matplotlib inline 
def imshow(img): 
    img = img/2 + 0.5 
    npimg = img.numpy() 
    plt.imshow(np.transpose(npimg, (1,2,0))) 
    plt.show() 
dataiter = iter(testloader) 
images, labels = dataiter.next() 
imshow(torchvision.utils.make_grid(images)) 
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


