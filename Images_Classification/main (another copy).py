import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os, copy, time
from multiprocessing import Process, freeze_support
# pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pylab import rcParams

#%matplotlib inline
#%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main():
    print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
    print(torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}, count = {torch.cuda.device_count()}, device_name = {torch.cuda.get_device_name(0)}')
    path = "/home/max/Projects/Images_Classification"
    path_folder = os.path.join(path,'models')
    model_path = os.path.join(path_folder, 'Dataset_20042020_ttv_v4_CUT.pt')
    data_dir = os.path.join(path, "Dataset_20042020_ttv_CUT")
    test_path = os.path.join(data_dir, "test")

    # Data augmentation and normalization for training Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
    	    #transforms.RandomRotation(15),
            #transforms.RandomResizedCrop(512),
            transforms.Resize(size=(224,224), interpolation=3),
            #transforms.Resize(224),
            #transforms.Grayscale(num_output_channels=3),
            #transforms.RandomGrayscale(p=0.3),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(size=(224,224), interpolation=3),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def visualize_model(model, num_images=7):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)



    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    inputs, classes = next(iter(dataloaders['train']))  # Get a batch of training data
    out = torchvision.utils.make_grid(inputs)
    train_dir = os.path.join(data_dir, "train")
    classes_counts = list([(folder, len(os.listdir(os.path.join(train_dir, folder)))) for folder in os.listdir(train_dir)])
    print(f'classnames = {class_names}')
    counts = list([class_count[1] for class_count in classes_counts])
    print(f'counts = {counts}')
    all_data_count = sum(counts)
    print(f'Count of all photos = {all_data_count}')
    max_of_counts_dataset = max(counts)
    div_func = lambda class_count: max_of_counts_dataset / class_count  # Max(Number of occurrences in most common class ) / (Number of occurrences in rare classes)
    classes_weights = list(map(div_func, counts))   # div_func = lambda class_count: 1 / class_count # classes_weights = list(map(div_func, counts))
    print(f'classes_weights = {classes_weights}')

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        list_of_losses = {'train': [], 'val': []}
        list_of_accuracies = {'train': [], 'val': []}
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                list_of_losses[phase].append(epoch_loss)
                list_of_accuracies[phase].append(float(epoch_acc))
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, list_of_losses, list_of_accuracies

    model_ft = models.resnet18(pretrained=True, progress=True)
    num_ftrs = model_ft.fc.in_features
    print(f'num_ftrs = {num_ftrs}')
    for name, child in model_ft.named_children():
        if (name in ['fc', 'layer3', 'layer4']) or ('bn' in name):  #
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False
    print(f'len(class_names) = {len(class_names)}')
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    print(f'classes_weights = {classes_weights}')
    # class_weights = torch.FloatTensor(classes_weights).cuda()
    # weight=class_weights
    criterion = nn.CrossEntropyLoss()
    print(f'model_ft.parameters() = {model_ft.parameters()}')
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, nesterov=False)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1,
                                           last_epoch=-1)  # step_size=7, gamma=0.2)
    plt.hist(classes_weights)
    plt.plot(list(range(len(classes_weights))), classes_weights)
    num_epochs = 25
    model_ft, list_of_losses, list_of_accuracies = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                               num_epochs=num_epochs)  # ignore_index=-1
    print(f'list_of_losses = {list_of_losses}')
    print(f'list_of_accuracies = {list_of_accuracies}')
    torch.save(model_ft, model_path)
    print(f'The model is saved as: {model_path}')
    data = {'epochs': list(range(1, num_epochs + 1)),
            'train_losses': list_of_losses['train'],
            'val_losses': list_of_losses['val'],
            'train_accs': list_of_accuracies['train'],
            'val_accs': list_of_accuracies['val']}
    losses_df = pd.DataFrame(data=data, columns={'epochs': 'epochs',
                                                 'train_losses': 'train_losses',
                                                 'val_losses': 'val_losses',
                                                 'train_accs': 'train_accs',
                                                 'val_accs': 'val_accs'})
    losses_df.to_csv(os.path.join(path, 'losses_df.csv'))
    plt.plot(list(range(len(list_of_losses['train']))), list_of_losses['train'])
    visualize_model(model_ft, num_images=6)

if __name__ == '__main__':
    freeze_support()
    Process(target=main).start()