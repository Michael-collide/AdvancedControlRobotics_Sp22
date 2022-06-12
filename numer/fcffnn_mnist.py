# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:25:56 2021

@author: Vik Gupta
"""

# Import python libraries that we will need
#import numpy as np
import torch
#import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np


device = 'cpu' # Change this to 'cuda' if torch.cuda.is_available()
# torch.cuda will enable parallel computations, which will make your code
# significantly faster. It requires nvidia GPU


# Download MNIST data, it consists of handwritten numbers
# as grey-scale images.

# This dataset is available online from LeCun's website and can be downloaded
# directly.

#trainset = datasets.MNIST(root='dataset/', download=True, train=True,
#                          transform=transforms.ToTensor())
#valset = datasets.MNIST(root='dataset/', download=True, train=False,
#                          transform=transforms.ToTensor())

# If the server is unavailable then you may get an error. In that case,
# we can just use data from other resources. We can use this previously
# downloaded data


trainset = datasets.MNIST(root='dataset/', download=False, train=True,
                          transform=transforms.ToTensor())
valset = datasets.MNIST(root='dataset/', download=False, train=False,
                          transform=transforms.ToTensor())


# We have training set, which we use to optimize the parameters (i.e. weights
# and biases)
# We have validation set, which we use to optimize hyperparameters (i.e. 
# learning rate, number of hidden layers, size of the layer, optimization
# method, loss function, regularization/data-normalization, mini-batch size)

# If want a transformation of data
# transform = transforms.Compose([transforms.ToTensor(),
#                      transforms.Normalize((0.5,), (0.5,)),])
# trainset = datasets.MNIST(root='dataset/', download=False, train=True,
#                           transform=transform)
# valset = datasets.MNIST(root='dataset/', download=False, train=False,
#                           transform=transform)

batch_size  = 32 # It is a hyperparameter

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader   = DataLoader(valset, batch_size=batch_size, shuffle=False)


# DataLoader supports automatically collating individual fetched data samples
# into batches.

# batch_size = number of examples (batch gradient-descent)
# batch_size = 1 (stochastic gradient-descent)
# batch_dize = anything in between (mini-batch gradient-descent)

# shuffle (=True) shuffles the images in the mini-batches in every iteration.


# Can check the data and DataLoader. Dataloader will give batch_size = 16 
# images for trainloader and 32 images for valloader 
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape) # (no. of images)x(no. of colour channels)x(pixels) 
                    # = batch_size x 1 x 28 x 28
print(labels.shape) # no. of labels should be equal to no. of images

# Let's see a few image
figure = plt.figure()
num_of_images = 30
for index in range(1,1+num_of_images):
    plt.subplot(3, 10, index)
    plt.axis('off')
    plt.imshow(images[index-1].numpy().squeeze(), cmap='gray_r')



# Build a neural network
input_size  = 784  # This is 28x28 and should be fixed
num_classes = 10   # This is to classify the numbers from 0 to 9, i.e. in 10 
                   # classes. It is fixed.
                   
# Hyperparameters
learning_rate = 0.001
num_epochs = 20
num_neurons = 20

# Creare Fully Connected Network
# It is a shallow network with only one hidden layer. Add one or more hidden
# layers to make it deep. You may also play with the number of features in
# hidden layers - These are two hyperparameters.
# You can try to normalize outputs from the individual layers - that is
# another hyperparameter that can be checked. 
class FirstNNet(nn.Module):
    def __init__(self, input_size, num_classes, num_neurons):
        super(FirstNNet,self).__init__() # same as super().__init__()
        self.fc1 = nn.Linear(input_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons,num_classes)
        
    def forward(self, x):
        x = nn.functional.sigmoid(self.fc1(x))
        x = nn.functional.log_softmax(self.fc2(x))
        # log of softmax is taken instead of softmax. This is for numerical
        # efficiency
        return x

learning_rates = np.arange(0.001, 0.006, 0.001)
neurons_number = np.arange(15, 36, 2)
# epochs = np.array([20, 30])
epochs = np.array([20])

parameters = [(lr, ne_num, e) for lr in learning_rates for ne_num in neurons_number for e in epochs]
result = []

for params in parameters:
    print("-----------------------------------")
    # Initialize network
    model = FirstNNet(input_size=input_size,num_classes=num_classes, num_neurons=params[1]).to(device)

    # Loss and Optimizer
    criterion = nn.NLLLoss()   # log_softmax + NLLLoss = softmax + cross_entropy
                               # loss that we learned in the class
    optimizer = optim.Adam(model.parameters(), lr = params[0])
    # optimizer, its type and parameters are hyperparameters.


    time0 = time()

    for e in range(params[2]):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).to(device)
            labels = labels.to(device)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss   = criterion(output, labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


    images, labels = next(iter(valloader))

    img = images[0].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    figure = plt.figure()
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


    correct_count, all_count = 0, 0
    for images,labels in valloader:
      for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))
    result.append([params[0], params[1], (correct_count/all_count)])
    print(result)

res = np.array(result)
print(res.shape)
np.savez("./result.npz", res)
