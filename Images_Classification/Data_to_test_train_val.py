import os           # !nvcc --version
import random       # import tensorflow # 2.2.5
import shutil       # #print(tensorflow.__version__)
from tqdm import tqdm

path = os.getcwd()
data_path = os.path.join(path, "Dataset_17072020")
data_ttv_path = os.path.join(path, "Dataset_17072020_ttv_not_cutted")

class Next_name:
    # Класс-итератор для выбора случайного файла из списка (shuffle_images)
    def __init__(self, list_of_items, start=0):
        self.num = start
        random.shuffle(list_of_items)
        self.list_of_items = list_of_items

    def __iter__(self):
        return self

    def __next__(self):
        item = self.list_of_items[self.num]
        self.num += 1
        return item

# Создадим папку Data
names = {}
if not os.path.exists(data_ttv_path):
    os.mkdir(data_ttv_path)

# переменные хранят пути к папкам [train, test, val]
train_path = os.path.join(data_ttv_path, "train")
test_path = os.path.join(data_ttv_path, "test")
val_path = os.path.join(data_ttv_path, "val")

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
    elif i <= round(0.95 * quantity):
        folder_to = os.path.join(val_path, folder_name)
    else:
        folder_to = os.path.join(test_path, folder_name)
    try:
        image_name = next(shuffled_images)
        image_from_path = os.path.join(folder_from, image_name)
        if (os.path.exists(image_from_path)) and (os.path.getsize(image_from_path) > 0):
            #print(f'os.path.join(folder_to, image_name) = {os.path.join(folder_to, image_name)}')
            shutil.copy(image_from_path, os.path.join(folder_to, image_name))
        else:
            print(f'Ошибка! Файл {image_from_path} битый или не существует!')
            print(f'Размер файла = {os.path.getsize(image_from_path)}')
    except Exception as inst:
        print(inst)


def make_test_train_again():
    # Функция копирует картинки по папкам train/test/val
    for i in tqdm(range(len(os.listdir(data_path)))):
        folder_name = os.listdir(data_path)[i]
        folder_from = os.path.join(data_path, folder_name)
        files_in_folder = os.listdir(folder_from)
        quantity =len(os.listdir(folder_from))
        shuffled_images = Next_name(os.listdir(folder_from))
        for i in tqdm(range(len(files_in_folder))):
            copy_file(i, quantity, shuffled_images, folder_name)

# Эта функция копирует файлы по папкам train/test/val
make_test_train_again()