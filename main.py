"""
'''1) Туристы отправились в пеший поход. На карте начерчена линия, по которой будут передвигаться путешественники. Святослав хочет присоединиться к группе в точке (х, у). Определите, находится ли эта точка на маршруте или нет.
В программе уже записан ввод двух чисел— x и y, каждое с новой строки, координаты точки, в которой Святослав хочет присоединиться к группе, а так же вызвана функция. Дополни код - допиши функцию, которая проверяет, попадёт ли эта точка на фиолетовую прямую. Если да — вывести “Попал!”, иначе — вывести “Не попал!”.'''
print('Задача 1')
def coordinate(x, y):
  if y == 2 and  -5 <= x <= 5:
    print('УРАААА, ОНи встретились :)')
  else:
    print('Святослав опять заблудился...')
'''
a = 3
b = 2
coordinate(a, b)
'''
#print('Задача 2')
'''2) В геолокации для кораблей указывается два числа: x градусов широты и y градусов долготы. Эти числа образуют координаты точки на карте. Давай поможем морякам и упростим им задачу расчёта расстояния между двумя такими точками.
В программе уже записан ввод четырёх чисел (x1, y1, x2, y2) и вызвана функция. Дополни код - напиши функцию, которая должна вычислить расстояние между ними и вывести ответ на экран.'''
'''
def count(x1, y1, x2, y2):
  result = (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 1/2
  return result
a = 5#float(input())
b = 5#float(input())
c = 7#float(input())
d = 7#float(input())
count(a, b, c, d)
print(str(count(a, b, c, d)))
'''
print('Задача 3')
'''3) На одном из каналов проводится телевикторина «Я всё знаю». Участники отвечают на вопросы и получают за правильные ответы баллы. В конце игры результаты вводятся в строку через запятую в следующем формате: имя – балл.
Напиши две функции: первая находит все числа в строке, записывает их в список и выводит его на экран, а вторая - находит среди чисел списка самое наибольшее и выводит его на экран в формате "Максимальный балл = (цифра)"'''
'''
list_of_names = ['Николай','Иван','Юля','Лариса']
iterator = iter(list_of_names)
list_a = ([int(input(f'Введите количество баллов участника {next(iterator)}: \n')) for i in range(4)])
print("Максимальный балл:", max(list_a))

print('Задача 4')
4) Ещё не забыл программу "математический помощник"? Пришло время дополнить его новыми функциями!
Напиши две функции: первая вычисляет и выводит на экран количество десятков в числе, а вторая - единиц в формате, как показано на картинке. Числа вводятся пользователем с клавиатуры и передаются в функции в качестве параметров.
'''
def count(a):
  
  if a > 9:
    print("Единицы:", a % 10 )# a - ((a//10)*10)) 
    print("Десятки:", (a % 100) // 10 )# (a - ((a//100)*100)-(a - ((a//10)*10)))//10)
  else:
    print("Единицы:", a)
    print("Десятки:", 0)

count(875789791)

print('Задача 5')
'''
'''5) Реализуй программу, которая была разобрана в теоретической части урока.
Пользователь дважды вводит по два целых числа. В программе уже записано начало объявления функции. Необходимо дописать функцию, чтобы она выводила наибольшее из чисел в двух запросах, а затем сложить результаты и вывести их на экран.
def comparison (num1, num2):'''
'''
def sravn(list1):      # a, b): # list1 = [a, b]
  return max(list1)

a = int(input())
b = int(input())
max_l = lambda list_: max(list_)
# max_l = lambda x, y: (x**2 + y**2)**0.5
print(sravn([a, b]))
d = int(input())
g = int(input())
print(sravn([d, g]))
print(max_l([a, b])+ max_l([d, g]))
print(sravn([a, b]) + sravn([d, g]))
"""
'''
Класс содержит функцию-итератор,
которая вызывается через лямбда-функцию
для итерации диалога - то есть, диалоги не должны 
повторяться
'''

class GEN_phrase():
    
  #next_lambda = lambda x: next(itr)
  #next_lambda = lambda x: print(next(self.itr))
  
  dict_of_phrases = {}
  def __init__(self, list_of_phrases):
    self.list_of_phrases = list_of_phrases
    self.itr = iter(list_of_phrases)
    self.schetchik = len(self.list_of_phrases)

  def next_phrase(self):
    #nextt = lambda x: next(self.itr)
    # next_lambda = lambda: next(self.itr)
    # print(next_lambda())
    if self.schetchik >= 0 :
      print((lambda: next(self.itr))())
      self.schetchik -= 1
    else:
        self.itr = iter(self.list_of_phrases)
        self.schetchik = len(self.list_of_phrases)

  def fill_phrases(self):
    for i in range(len(self.list_of_phrases)):
      self.dict_of_phrases[i] = self.list_of_phrases[i]
  

list_of_phrases = ['фраза 0', 'фраза 1', 'фраза 2','фраза 3', 'любая фраааза','еще фраза']
player1 = GEN_phrase(list_of_phrases)
player1.fill_phrases()
player1.next_phrase()
player1.next_phrase()
player1.next_phrase()
player1.next_phrase()
player1.next_phrase()
player1.next_phrase()
player1.next_phrase()
#itr = iter(player1.list_of_phrases)
#[print(player1.next_phrase()) for _ in range(len(player1.list_of_phrases))]
#[print(player1.next_lambda()) for _ in range(len(player1.list_of_phrases))]


'''
# key - index, meaning - phrase
# dict_of_phrases[key] = meaning
list_of_phrases = ['фраза 0', 'фраза 1', 'фраза 2']
dict_of_phrases = {}
for i in range(len(list_of_phrases)):
    dict_of_phrases[i] = list_of_phrases[i]
print(dict_of_phrases)
'''