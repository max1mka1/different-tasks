from __future__ import print_function, division
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from torch.autograd import Variable






plt.ion()

path = os.getcwd()
num_epochs = 50
models_path = "/home/max/WORK/Images_Classification/models"
dataset_name = 'OnlyAug_10k'
data_dir = os.path.join(path, dataset_name) # './hymenoptera_data'


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


image_datasets = ImageFolder(data_dir, transform=Compose([Resize((299, 299)), ToTensor()]))
print(len(image_datasets))
datasets_1 = train_val_dataset(image_datasets)
dataloaders = {x: DataLoader(datasets_1[x], 32, shuffle=True, num_workers=4) for x in ['train', 'val']}
x, y = next(iter(dataloaders['train']))
print(len(dataloaders['train']))
print(len(dataloaders['val']))
# The original dataset is available in the Subset class
print(datasets_1['train'].dataset)
# Результирующий размер картинок определяется трансформациями
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

'''
# Сам объект датасета
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x])
    for x in ['train', 'val']
}

# специальный класс для загрузки данных в виде батчей
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    for x in ['train', 'val']
}
'''


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

dataset = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))


class_names = dataset.classes
print(class_names)

print(len(dataset))
datasets = train_val_dataset(dataset)
print(len(datasets['train']))
print(len(datasets['val']))
# The original dataset is available in the Subset class
print(datasets['train'].dataset)

dataloaders = {x: DataLoader(datasets[x], 32, shuffle=True, num_workers=4) for x in ['train', 'val']}
x, y = next(iter(dataloaders['train']))
model = models.googlenet(pretrained=True)
# num_features -- это размерность вектора фич, поступающего на вход FC-слою
num_features = 1024
# Заменяем Fully-Connected слой на наш линейный классификатор
model.fc = nn.Linear(num_features, 7)

use_gpu = ['cuda:0' if torch.cuda.is_available() else 'cpu']
print(f"use_gpu = {use_gpu}")

# Использовать ли GPU
if use_gpu:
    model = model.cuda()

dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15, 12))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model(model, num_images=6):
    images_so_far = 0
    # fig = plt.figure()
    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


def evaluate(model):
    model.train(False)

    runninig_correct = 0
    for data in dataloaders['val']:
        # получаем картинки и метки
        inputs, labels = data

        # переносим на GPU, если возможно
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward pass
        output = model(inputs)
        _, predicted = torch.max(output, 1)

        runninig_correct += int(torch.sum(predicted == labels))

    return runninig_correct / dataset_sizes['val']


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    losses = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # каждя эпоха имеет обучающую и тестовую стадии
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # установаить модель в режим обучения
            else:
                model.train(False)  # установить модель в режим предсказания

            running_loss = 0.0
            running_corrects = 0

            # итерируемся по батчам
            for data in dataloaders[phase]:
                # получаем картинки и метки
                inputs, labels = data

                # оборачиваем в переменные
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = inputs, labels

                # инициализируем градиенты параметров
                optimizer.zero_grad()

                # forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward pass + оптимизируем только если это стадия обучения
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # статистика
                running_loss += loss.item()
                running_corrects += int(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step(epoch_loss)
                model.train(True)  # установаить модель в режим обучения
            else:
                model.train(False)  # установить модель в режим предсказания

            losses[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # если достиглось лучшее качество, то запомним веса модели
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # загрузим лучшие веса модели
    model.load_state_dict(best_model_wts)
    return model, losses


# В качестве cost function используем кросс-энтропию
loss_fn = nn.CrossEntropyLoss()
# В качестве оптимизатора - стохастический градиентный спуск
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
# Умножает learning_rate на 0.1 каждые 7 эпох (это одна из эвристик, не было на лекциях)
# exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='max', factor=0.1, patience=0, verbose=True)
exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, threshold=0.01,
                                                              patience=5, verbose=True)

model.aux_logits = False # (model, criterion, optimizer, scheduler, num_epochs=25):
model, losses = train_model(model, loss_fn, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

figure = plt.figure(figsize=(12, 7))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(losses['train'], label='train')
plt.plot(losses['val'], label='val')
plt.legend();

models_dataset_path = os.path.join(models_path, dataset_name)
if not os.path.exists(models_dataset_path):
    os.mkdir(models_dataset_path)
torch.save(model.state_dict(), os.path.join(models_dataset_path, 'GoogleNet_FT' + dataset_name + '.pth'))

model.load_state_dict(torch.load(os.path.join(models_path, dataset_name, 'GoogleNet_FT' + dataset_name + '.pth')))

print("Accuracy: {0:.4f}".format(evaluate(model)))