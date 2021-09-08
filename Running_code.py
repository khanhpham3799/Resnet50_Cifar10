!pip install torchsummary
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader, TensorDataset # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import skimage
from skimage import io
from torchsummary import summary
from tqdm.notebook import tqdm
from torchvision.transforms.transforms import ToPILImage
import argparse
import torch.backends.cudnn as cudnn
from Resnet_Unet_model import ResNet50

#data transform
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(p = 0.01),
    #transforms.RandomAffine((-10,10)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='CIFAR10_train', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='CIFAR10_test', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

#training and testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

net = ResNet50()
net.to(device)
# summary(net,(3,32,32))

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=1)


def load_checkpoint(checkpoint, model, optimizer, load = False):
    if load:
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])

# Try load checkpoint
load_checkpoint(torch.load("Resnet50_checkpoint.pth"), model=net, optimizer=optimizer, load=True)
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.001

# Training
def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(total=len(trainloader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for param_group in optimizer.param_groups:
                new_lr = param_group['lr']
            pbar.update(inputs.shape[0])
            pbar.set_postfix({'batch_index': batch_idx})
            pbar.set_description("Loss: %.3f | Acc: %.3f (%d | %d) | lr: %f"
                                 % ((train_loss) / (batch_idx + 1), 100 * (correct / total), correct, total, new_lr)
                                 )
    train_acc = 100 * (correct / total)
    return train_acc

best_acc = torch.load('Resnet50_bestacc.pth')
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}', unit='img') as pbar:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.update(inputs.shape[0])
                pbar.set_postfix({'batch_index': batch_idx})
                pbar.set_description("Test_Loss: %.3f | Tets_Acc: %.3f (%d | %d)"
                                     % ((test_loss) / (batch_idx + 1), 100 * (correct / total), correct, total)
                                     )

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': acc
        }
        torch.save(acc, 'Resnet50_bestacc.pth')
        torch.save(state, 'Resnet50_checkpoint.pth')
        best_acc = acc
    return acc


start_epoch = 0
i = 0
for epoch in range(start_epoch, start_epoch + 150):
    train_acc = train(epoch, trainloader=trainloader)
    test_acc = test(epoch)
    scheduler.step(test_acc)
    if test_acc < best_acc:
        i += 1
    if test_acc >= best_acc:
        i = 0
    if train_acc > 99 and i > 30:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine((-20, 20)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='CIFAR10_train', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        load_checkpoint(torch.load("Resnet50_checkpoint.pth"), model=net, optimizer=optimizer)
        best_acc = torch.load('Resnet50_bestacc.pth')
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        i = 0