import torch,torchvision
import numpy as np
from torchvision import transforms,datasets
from time import time
import sys
import layer
import NN

use_cuda=True

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1,32,kernel_size=5,padding=2,stride=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = torch.nn.Conv2d(32,64,kernel_size=5,padding=2,stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.dense1 = torch.nn.Linear(7*7*64,1024)
        self.dense2 = torch.nn.Linear(1024,10)
        self.precision = 0.
        self.epoch = 0

    def forward(self,input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.pool1(input)
        input = self.conv2(input)
        input = self.relu(input)
        input = self.pool2(input)
        input = input.view(len(input),-1)
        input = self.dense1(input)
        input = self.relu(input)
        input = self.dense2(input)
        return input

    def crypted_forward(self,HE,keys):
        pass

class CNN_modified(torch.nn.Module):
    def __init__(self):
        super(CNN_modified,self).__init__()
        self.relu = layer.relu(2)
        self.conv1 = layer.Conv(1,32,kernel_size=5,padding=2,stride=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = layer.Conv(32,64,kernel_size=5,padding=2,stride=1)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2,stride=2)
        self.dense1 = layer.Dense(7*7*64,1024)
        self.dense2 = layer.Dense(1024,10)
        self.precision = 0.
        self.epoch = 0

    def forward(self,input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.pool1(input)
        input = self.conv2(input)
        input = self.relu(input)
        input = self.pool2(input)
        input = input.view(len(input),-1)
        input = self.dense1(input)
        input = self.relu(input)
        input = self.dense2(input)
        return input

    def crypted_forward(self,HE,keys):
        pass

def train(model,train_dataloader,optimizer,use_cuda):
    if use_cuda:
        for img, label in train_dataloader:
            img = img.cuda()
            label = label.cuda()
            predict = model(img)
            loss = loss_func(predict,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for img, label in train_dataloader:
            predict = model(img)
            loss = loss_func(predict,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def validate(model,test_dataloader,use_cuda):
    correct = 0
    total = 0
    if use_cuda:
        for img, label in test_dataloader:
            img = img.cuda()
            label = label.cuda()
            predict = model(img)
            total += len(label)
            correct += torch.sum(predict.argmax(-1)==label).item()
    else:
        for img, label in test_dataloader:
            predict = model(img)
            total += len(label)
            correct += torch.sum(predict.argmax(-1)==label).item()
    precision = correct/total*100
    return precision

def encrypted_validate(model,test_dataloader,group,HE,T,scale,print_every=None,use_cuda=True):
    model.double()
    model.build(HE,T,scale,group)
    S = HE.getSecretkey(T)
    I = torch.eye(len(T),dtype=T.dtype,device=T.device)
    M_encrypt = HE.keySwitchMatrix(I,T)
    correct = 0
    total = 0
    count = 0
    if use_cuda:
        for img, label in test_dataloader:
            img = img.cuda().double()
            imgg = scale*img
            shape = imgg.shape
            imgg = imgg.view(shape[0]//group,-1)
            c = HE.keySwitch(M=M_encrypt,c=imgg)
            c = c.view(shape[0]+group,*shape[1:])
            label = label.cuda()
            predict = model.crypted_forward(HE=HE,input=c)
            predict = predict.view(shape[0]//group+1,-1)
            predict = HE.decrypt(S=S,c=predict)/scale
            predict = predict.view(shape[0],-1)
            total += len(label)
            correct += torch.sum(predict.argmax(-1)==label).item()
            count += 1
            if print_every and count%print_every==0:
                print('iter {}, correct = {}, total = {}'.format(count,correct,total))
    else:
        pass
    precision = correct/total*100
    return precision



transform = transforms.Compose([
    transforms.RandomCrop(size=(28,28),padding=4),
    transforms.ToTensor()
    ])

mnist_train = datasets.MNIST(
    root='../datasets/MNIST/',
    train=True,
    transform=transform,
    download=True
    )

mnist_test = datasets.MNIST(
    root='../datasets/MNIST/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
    )
def TestDataloader(batch_size):
    return torch.utils.data.DataLoader(
        mnist_test,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)

train_dataloader = torch.utils.data.DataLoader(
        mnist_train,batch_size=128,shuffle=True,num_workers=0,drop_last=False)

test_dataloader = torch.utils.data.DataLoader(
        mnist_test,batch_size=128,shuffle=True,num_workers=0,drop_last=False)

loss_func = torch.nn.CrossEntropyLoss()

if __name__=='__main__':

    if len(sys.argv)>1:
        model = torch.load(sys.argv[1])
    else:
        model = NN.AlanNet(l=2)
    if use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=10, 
        verbose=True, 
        threshold=0.0001, 
        threshold_mode='rel', 
        cooldown=0, 
        min_lr=1e-6, 
        eps=1e-08)
    

    start = model.epoch
    if use_cuda:
        for epoch in range(start,300):
            model.lr.append(optimizer.param_groups[0]['lr'])
            t1 = time()
            train(model,train_dataloader,optimizer,True)
            t2 = time()
            precision = validate(model,test_dataloader,use_cuda=True)
            t3 = time()
            scheduler.step(precision)
            print('epoch {:2}, lr = {} precision = {:4.4}%,\n correct = {:4}, total = {:4}'.format(epoch+1,optimizer.param_groups[0]['lr'],precision,correct,total))
            print('training time = {}s, test time = {}s'.format(t2-t1,t3-t2))
            print()
            model.epoch += 1
            model.precision.append(precision)
            if precision > model.precision[-2]:
                model.precision = precision
                torch.save(model,'cryptoNN-l=2.pth')