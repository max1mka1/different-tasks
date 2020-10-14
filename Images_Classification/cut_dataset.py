import os
from random import randint, shuffle
import shutil
from tqdm import tqdm

path = os.getcwd()
data_path = os.path.join(path, 'Dataset_17072020')
dataset_name = 'Dataset_17072020_ttv_CUT'


class Next_name:
    # Класс-итератор для выбора случайного файла из списка (shuffle_images)
    def __init__(self, list_of_items, start=0):
        self.num = start
        shuffle(list_of_items)
        self.list_of_items = list_of_items

    def __iter__(self):
        return self

    def __next__(self):
        item = self.list_of_items[self.num]
        self.num += 1
        return item




def get_counts_from_test_train_val(data_path):
    for folder in os.listdir(data_path):
        folder_name = os.path.join(data_path, folder)
        print(f'folder_name = {folder_name}')
        for folder_ in os.listdir(folder_name):
            folder_name_ = os.path.join(folder_name, folder_)
            print(f'folder_name_ = {folder_name_}')
            print(f'Folder: {folder_name_}, count = {len(os.listdir(folder_name_))}')


def get_counts_from_data(data_path):
    min_dataset = 9999
    for folder in os.listdir(data_path):
        folder_name = os.path.join(data_path, folder)
        print(f'folder_name = {folder}')
        count = len(os.listdir(folder_name))
        print(f'Folder: {folder_name}, count = {count}')
        if count < min_dataset:
            min_dataset = count
    return min_dataset


def create_folders(dataset_name=dataset_name):
    data_path = os.path.join(path, dataset_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)


def get_counts_from_data(data_path):
    min_dataset = 9999
    for folder in os.listdir(data_path):
        folder_name = os.path.join(data_path, folder)
        print(f'folder_name = {folder}')
        count = len(os.listdir(folder_name))
        print(f'Folder: {folder_name}, count = {count}')
        if count < min_dataset:
            min_dataset = count
    return min_dataset

min_data_count = get_counts_from_data(data_path)
folder_names = os.listdir(data_path)
print(f'min_data_count = {min_data_count}')
print(f'folder_names = {folder_names}')

def get_counts_from_data(data_path):
    min_dataset = 9999
    for folder in os.listdir(data_path):
        folder_name = os.path.join(data_path, folder)
        print(f'folder_name = {folder}')
        count = len(os.listdir(folder_name))
        print(f'Folder: {folder_name}, count = {count}')
        if count < min_dataset:
            min_dataset = count
    return min_dataset

# переменные хранят пути к папкам [train, test, val]
create_folders(dataset_name=dataset_name)
new_dataset_path = os.path.join(path, dataset_name)
train_path = os.path.join(new_dataset_path, "train")
test_path = os.path.join(new_dataset_path, "test")
val_path = os.path.join(new_dataset_path, "val")

# создадим папки [train, test, val]
list_of_train_test_val = [train_path, test_path, val_path]
for path in list_of_train_test_val:
    if not os.path.exists(path):
        os.mkdir(path)
    for folder in os.listdir(data_path):
        cur_path = os.path.join(path, folder)
        print(f'cur_path = {cur_path}')
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)


def copy_file(i, quantity, shuffled_images, folder_name):
    # Копирует файл из папки Data в папку train/test/val
    folder_from = os.path.join(data_path, folder_name)
    folder_to = ''
    if i <= round(0.6 * quantity):
        folder_to = os.path.join(train_path, folder_name)
    elif i <= round(0.9 * quantity):
        folder_to = os.path.join(val_path, folder_name)
    else:
        folder_to = os.path.join(test_path, folder_name)
    # folder_to = lambda x: train_folder if x < round(0.9 * q) else test_folder # if x <round(0.9 * q) else val_folder
    try:
        image_name = next(shuffled_images)
        #print(f'image_name = {image_name}')
        image_from_path = os.path.join(folder_from, image_name)
        #print(f'image_from_path = {image_from_path}')
        if (os.path.exists(image_from_path)) and (os.path.getsize(image_from_path) > 0):
            #print(f'os.path.join(folder_to, image_name) = {os.path.join(folder_to, image_name)}')
            shutil.copy(image_from_path, os.path.join(folder_to, image_name))
        else:
            print(f'Ошибка! Файл {image_from_path} битый или не существует!')
            print(f'Размер файла = {os.path.getsize(image_from_path)}')
    except Exception as inst:
        print(inst)


def make_test_train_again(min_data_count):
    # Функция копирует картинки по папкам train/test/val
    for i in tqdm(range(len(os.listdir(data_path)))):
        folder_name = os.listdir(data_path)[i]
        #print(folder_name)
        folder_from = os.path.join(data_path, folder_name)
        #print(folder_from)
        #print(files_in_folder)
        #print(f'Папка: {folder_name}')
        #print(f'В папке: {len(os.listdir(folder_from))} файлов!')
        quantity = min_data_count
        shuffled_images = Next_name(os.listdir(folder_from))
        for i in tqdm(range(quantity)):
            copy_file(i, quantity, shuffled_images, folder_name)

# Эта функция копирует файлы по папкам train/test/val
make_test_train_again(min_data_count)