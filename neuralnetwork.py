import os
import torch
from torch.autograd import Variable
import time
import helper
import torch.utils.data as data_loader


# trains a given network with the dataloader and the iterations given
def train_network(net, dset_loader, testset, iterations, optimizer, criterion, device):
    testset_loader = data_loader.DataLoader(testset, shuffle=True, num_workers=4)
    net = net.to(device)
    for epoch in range(iterations):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dset_loader, 0):
            # get the inputs
            (inputs, classes), _ = data

            # wrap them in Variable
            inputs, classes = Variable(inputs), Variable(classes)

            # move them to gpu if accessible
            inputs, classes = inputs.to(device), classes.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net.forward(inputs)

            loss = criterion(outputs, classes)
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if epoch % 100 == 99:
            tester = helper.TestHelper(testset_loader, testset.classes, net, device)
            tester.test_total_precision()
            tester.print_total_precision("ResNet152", epoch+1)
            torch.save(net.state_dict(), os.getcwd() + '/serialized_nets/ResNet152SimilarPlaces_v' + str(epoch+1))
