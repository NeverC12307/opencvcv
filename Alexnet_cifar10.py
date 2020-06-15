import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


# define transform
# hint: Normalize(mean, var) to normalize RGB
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# define trainloader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# define testloader
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
# define class
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # constract network
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


net = Net()
print(net)
# define loss
cost = nn.CrossEntropyLoss()
# define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# iteration for training
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()

        # print loss result
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.001
print('done')

'''
#get random image and label
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('groundTruth: ', ''.join('%6s' %classes[labels[j]] for j in range(4)))
#get the predict result
outputs = net(Variable(images))
_, pred = torch.max(outputs.data, 1)
print('prediction: ', ''.join('%6s' %classes[labels[j]] for j in range(4)))
'''
# test the whole result
correct = 0.0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum()
print('average Accuracy: %d %%' % (100 * correct / total))

# list each class prediction
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, pred = torch.max(outputs.data, 1)
    c = (pred == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += float(c[i])
        class_total[label] += 1
print('each class accuracy: \n')
for i in range(10):
    print('Accuracy: %6s %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))