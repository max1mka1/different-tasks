import sys
import os
import io
import time
#import PIL
import selenium
import requests
#from PIL import Image
from time import sleep
from selenium import webdriver
import urllib
import urllib.request

DRIVER_PATH = 'D:\parser\drivers\chromedriver.exe'
wd = webdriver.Chrome(executable_path=DRIVER_PATH)
wd.maximize_window()   # For maximizing window
wd.implicitly_wait(3) # gives an implicit wait for 20 seconds
wd.get('https://google.com')
search_box = wd.find_element_by_css_selector('input.gLFyf')
search_box.send_keys('Dogs')


class SimpleIterator:
    def __iter__(self):
        return self

    def __init__(self, limit):
        self.limit = limit
        self.counter = 0

    def __next__(self):
        if self.counter < self.limit:
            self.counter += 1
            return self.counter
        else:
            raise StopIteration

s_iter1 = SimpleIterator(500)

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(sleep_between_interactions)

    # build the google query
    search_url = """https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"""

    # load the page
    wd.get(search_url.format(q=query))
    sleep(0.7)
    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.rg_i")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.rg_i')
            for actual_image in actual_images:
                if actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
            else:
                print("Found:", len(image_urls), "image links, looking for more ...")
                sleep(0.8)
                load_more_button = wd.find_element_by_class_name(".ksb")
                #load_more_button = wd.find_element_by_id('n3VNCb')
                # load_more_button = wd.find_element_by_css_selector("")  #
                if load_more_button:
                    wd.execute_script("document.querySelector('.ksb').click();")  # .ksb

            # move the result startpoint further down
            results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path:str,url:str):
    img_url = url
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")
    try:
        image_file = io.BytesIO(image_content)
        with urllib.request.urlopen(url) as url:
            s = url.read()
            print(f's = {s}')
            img_url = url
            img = s # urllib.urlopen(img_url).read()
            img_name = str(next(s_iter1)) + '.jpg'    # img_url.split('.')[len(img_url.split('.')) - 2] +  '.'+ img_url.split('.')[len(img_url.split('.')) - 1]
            # if os.path.isfile(img_name):
                # os.remove(img_name)
            f = open(folder_path + '/' + img_name, "wb")
            f.write(img)
            f.close()
            '''
            image = image.open(image_file).convert('RGB')    # Image
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=85)
            '''
            print(f"SUCCESS - saved {url} - as {folder_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def search_and_download(search_term:str,driver_path:str,target_path='./images',number_images=5):
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=2)

    for elem in res:
        persist_image(target_folder,elem)


search_term = 'столбчатая диаграмма'
target_path = './images'
number_images = 5

search_and_download(
    search_term=search_term,
    driver_path=DRIVER_PATH,
    target_path=target_path,
    number_images=number_images
)
