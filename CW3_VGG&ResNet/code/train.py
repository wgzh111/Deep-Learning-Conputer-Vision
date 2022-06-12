from sklearn.metrics import confusion_matrix
import itertools
from torchviz import make_dot
import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Union, List, Dict, Any, cast
import matplotlib.pyplot as plt
from VGG import vgg16
from resnet import resnet18


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


def train(md, dataset, num_class, channel, train_loader, test_loader,
          num_epoches=5,
          batch_size=256, plot=True):
    # select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define the model
    if md == "vgg":
        model = vgg16(True, num_class, channel).to(device)
    elif md == "resnet":
        model = resnet18(True, num_class, channel).to(device)

    if dataset == "mnist":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer=optim.Adam(model.parameters(),lr=0.1)
    train_loss = np.zeros(0)
    train_acc = np.zeros(0)
    test_loss = np.zeros(0)
    top1_acc = np.zeros(0)
    top5_acc = np.zeros(0)
    lossPimage = np.zeros(0)

    starttime = datetime.datetime.now()
    print("start time:", starttime)
    all_preds = torch.tensor([])
    all_label = torch.tensor([])
    all_preds = all_preds.to(device)
    all_label = all_label.to(device)
    for epoch in range(num_epoches):
        print('----------- epoch {} -----------'.format(epoch + 1))
        running_loss = 0.0
        running_acc = 0.0
        total_train = len(train_dataset)
        for i, data in enumerate(train_loader, 0):

            inputs, label = data
            # cuda
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                label = label.cuda()
            inputs = Variable(inputs)
            label = Variable(label)
            outputs = model(inputs)
            loss = criterion(outputs, label)

            lossPimage = np.append(lossPimage, loss.item() * label.size(0))

            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(outputs, 1)

            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()

            running_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Finish Training {} epoch, Loss: {:.6f}, Acc.: {:.6f}'.format(
            epoch + 1, running_loss / total_train, (running_acc / total_train)))
        train_loss = np.append(train_loss, running_loss / total_train)

        train_acc = np.append(train_acc, (running_acc / total_train))

        model.eval()
        eval_loss = 0
        eval_acc_top1 = 0
        eval_acc_top5 = 0
        total_test = len(test_dataset)

        for data in test_loader:
            inputs, label = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                label = label.cuda()
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, label)

                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(outputs, 1)
                num_correct = (pred == label).sum()
                eval_acc_top1 += num_correct.item()

                correct = 0
                maxk = max((1, 5))
                label_resize = label.view(-1, 1)
                _, preds = outputs.topk(maxk, 1, True, True)
                correct += torch.eq(preds, label_resize).sum()
                eval_acc_top5 += correct.item()
                if epoch == num_epoches - 1:
                    all_preds = torch.cat((all_preds, pred.cuda()), dim=0)
                    all_label = torch.cat((all_label, label), dim=0)
        print('Test Loss: {:.6f}, Top1_Acc: {:.6f} , Top5_Acc: {:.6f}'.format(
            eval_loss / total_test, eval_acc_top1 / total_test, eval_acc_top5 / total_test))
        test_loss = np.append(test_loss, eval_loss / total_test)
        top1_acc = np.append(top1_acc, eval_acc_top1 / total_test)
        top5_acc = np.append(top5_acc, eval_acc_top5 / total_test)

    endtime = datetime.datetime.now()
    print("End time:", endtime)
    # torch.save(model.state_dict(), '/model')

    if plot:
        cm = confusion_matrix(all_label.cpu(), all_preds.cpu())

        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(cm, classes)
    return train_loss, test_loss, lossPimage, train_acc, endtime



trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Pad(2),
     transforms.Normalize((0.5), (0.5))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans)
mnist_train = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=trans)
mnist_test = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


def plotreslut(md, num_epoches, train_loss, test_loss, lossPimage, train_acc, endtime):
    x = np.linspace(1, num_epoches, num_epoches)
    plt.title(md + '_loss')
    plt.plot(x, train_loss, color='blue', label='training loss')
    plt.plot(x, test_loss, color='red', label='testing loss')
    plt.legend(['train loss', 'test loss'])
    plt.show()

    plt.title(md + '_acc')
    plt.plot(x, train_acc, color='blue', label='training acc')
    plt.xlabel('epoch')
    plt.ylabel('Acc.')
    plt.show()

    # np.save('/content/data/{}{}.npy'.format(endtime,num_epoches),lossPimage)


print("VGG16__MNIST")
md = "vgg"
num_epoches = 5
train_loss, test_loss, lossPimage, train_acc, endtime = train(md, "mnist", 10, 1, mnist_train, mnist_test,
                                                              num_epoches=num_epoches)

print("resnet__MNIST")
md = "resnet"
num_epoches = 5
train_loss, test_loss, lossPimage, train_acc, endtime = train(md, "mnist", 10, 1, mnist_train, mnist_test,
                                                              num_epoches=num_epoches)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
cif10_train = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
cif10_test = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print("VGG16__CIFAR10")
md = "vgg"
num_epoches = 25
train_loss, test_loss, lossPimage, train_acc, endtime = train(md, "cifar10", 10, 3, cif10_train, cif10_test,
                                                              num_epoches=num_epoches)

plotreslut(md, num_epoches, train_loss, test_loss, lossPimage, train_acc, endtime)

Sample = torch.rand(1, 1, 224, 224)
network = vgg16(True, 10, 1)
VGG16_Architecture_plot = make_dot(network(Sample), params=dict(network.named_parameters()))
# displaying the VGG flow diagram
VGG16_Architecture_plot.render("vgg", format="png")
display(VGG16_Architecture_plot)


Sample = torch.rand(1, 1, 224, 224)
network = resnet18(True, 10, 1)
res_Architecture_plot = make_dot(network(Sample), params=dict(network.named_parameters()))
# displaying the VGG flow diagram
res_Architecture_plot.render("res", format="png")
display(res_Architecture_plot)

print("resnet__CIFAR10")
md = "resnet"
num_epoches = 25
train_loss, test_loss, lossPimage, train_acc, endtime = train(md, "cifar10", 10, 3, cif10_train, cif10_test,
                                                              num_epoches=num_epoches)

plotreslut(md, num_epoches, train_loss, test_loss, lossPimage, train_acc, endtime)

