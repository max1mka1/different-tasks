import os
import pandas as pd
import urllib.request
from tqdm import tqdm
from joblib import Parallel, delayed

treads = -1
path = os.getcwd() # "/content/drive/My Drive/Colab Notebooks/Psychotypes_detection/Neural Network/"
data_pathname = "Dataset_17072020"
data_path = os.path.join(path, data_pathname)
images_csv = os.path.join(path, "psycho_photo_dataset.csv")

dict_of_id = {1001: "Истероидный", 1002: "Эпилептоидный",
              1003: "Паранояльный", 1004: "Эмотивный",
              1005: "Шизоидный", 1006: "Гипертимный",
              1007: "Тревожный"}

pd_links = pd.read_csv(filepath_or_buffer=images_csv,
                       encoding="cp1251",
                       delimiter=";")


class Iterator:
    # Класс именует файлы .jpg по имени радикала и номеру файла
    def __init__(self, start=0):
        self.num = start

    def __iter__(self):
        return self

    def __next__(self):
        name = self.num
        self.num += 1
        return name


class Namer:
    # Класс именует файлы .jpg по имени радикала и номеру файла
    def __init__(self, name, start=0):
        self.num = start
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        name = self.name + "_" + str(self.num) + ".jpg"
        self.num += 1
        return name


names = {}
data_path = os.path.join(path, data_pathname)
if not os.path.exists(data_path):
    os.mkdir(data_path)


#Создадим папочки для каждого из радикалов
for ids, radical_folder in dict_of_id.items():
    names[ids] = Namer(radical_folder)
    folder = os.path.join(data_path, radical_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)


def Download_and_save(i):
    if pd_links['First class'][i] not in ['Плохое фото', 'Старт', 'Непонятно']:
        try:
            link = pd_links['URL'][i] # , id_ int(pd_links['id'][i])
            id_name = pd_links['First class'][i] # dict_of_id[id_]
            folder_path = os.path.join(data_path, id_name)
            file_path = os.path.join(folder_path, pd_links['First class'][i] + str(i) + '.jpg')
            with open(file_path,'wb') as file: # f = open(file_path,'wb')# f.write(urllib.request.urlopen(link).read()# f.close()
                file.write(urllib.request.urlopen(link).read())
        except:
            print('Exception_error!')
            pass
    else:
        pass


Parallel(n_jobs=-1)(delayed(Download_and_save)(i) for i in tqdm(range(pd_links.shape[0])))