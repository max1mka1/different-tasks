
import re
from bs4 import BeautifulSoup as BS
from requests import get

def get_link(topic):
    link = "https://ru.wikipedia.org/wiki/" + topic.capitalize()
    return link

def get_topic_page(topic):
    link = get_link(topic)
    html_content = get(link).text
    # with open("new.html", "w", encoding="utf-8") as f:
    #     f.write(html_content)
    return html_content


#функция ищет ссылки по странице, начинающиеся с /wiki/
def get_links(topic):
    html_content = get_topic_page(topic)
    soup = BS(html_content, 'html.parser')
    str = re.compile('^/wiki/', re.MULTILINE)
    li = soup.find_all('a', attrs={"href": str})
    li2 = [n.get('href') for n in li]
    return li2

def get_topic_words(topic, from_all_links = True):
    html_content = get_topic_page(topic)
    words = re.findall("[а-яА-Я\-\']{3,}", html_content)
    #если ищем слова по всем ссылкам, то добавляем в массив слов слова с других страниц
    if from_all_links:
        links = get_links(topic)
        for li in links:
# организуем ссылку, получаем текст со страницы и ищем слова
            link = "https://ru.wikipedia.org" + li
            html_content = get(link).text
            words += re.findall("[а-яА-Я\-\']{3,}", html_content)
    return words


def get_common_words(topic,from_all_links = True):
    words_list = get_topic_words(topic,from_all_links)
    rate = {}
    for word in words_list:
        if word in rate:
            rate[word] += 1
        else:
            rate[word] = 1
    rate_list = list(rate.items())
    rate_list.sort(key=lambda x: -x[1])
    print(rate_list)
    return rate_list


def visualize_common_words(topic, from_all_links = True):
    words = get_common_words(topic,from_all_links)
    for w in words[100:110]:
        print(w[0])


def main():
    topic = input("Topic: ")
    from_all_links = input('Искать слова по соседним страницам? True/False: ')
    visualize_common_words(topic,from_all_links)

if __name__ == '__main__':
    main()

