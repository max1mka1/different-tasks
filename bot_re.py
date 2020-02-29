from random import randint
import re
'''
def createGenerator():
    mylist = ['Прощай, создатель', 'Уже прощались. Забыл что ли?', 'eferrgrgtg']
    for i in mylist:
        yield i 
    i = -1
    while True:
        i += 1
        yield mylist[i % 3]
    while True:
        yield mylist[randint(0, 2)]


mygenerator = createGenerator()
i = 0
while i < 10:
    try:
        print(next(mygenerator))
    except StopIteration:
        print('StopIteration error executed!!!!')
    i += 1


a = int(input())
b = 5
i = 0
while i < 10:
    try:
        print(a/b)
    except ZeroDivisionError:
        print('ZeroDivisionError error executed!!!!')
    i += 1

try:
    print(a/0)
except ZeroDivisionError:
    print('ZeroDivisionError error executed!!!!')
'''
'''
при?ве?т
    Знак вопроса, ?, проверяющий наличие совпадения ноль или один раз. Например, home-?brew соответствует как homebrew, так и home-brew.
    Первый метасимвол для повторения это *. Он указывает, что предыдущий символ может быть сопоставлен ноль и более раз, вместо одного сравнения.
'''

p = re.compile('пр.в.т')
print(p)


def main():
    while True:
        try:
          bot.polling(none_stop=True, interval=0)
        except:
          print('bolt')
          logging.error(f'error: {sys.exc_info()[0]}')
          time.sleep(5)

if __name__ == '__main__':
    main()

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message)
    bot.send_message(message.chat.id, 'Привет! id чата ' + str(message.chat.id))

import logging
import sys
from time import sleep

    bot.send_message(message.chat.id,
                     'Your id = ' + str(message.from_user.id) + "\n" +
                     'Your username = ' + str(message.from_user.username) + "\n" +
                     'Your first_name = ' + str(message.from_user.first_name) + "\n" +
                     'Your last_name = ' + str(message.from_user.last_name))