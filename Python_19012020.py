''' Перед тобой программа, в которой описан класс «Прямоугольник». В нём есть четыре свойства: координата угла x, координата угла y, ширина и высота прямоугольника.
В программе создано два экземпляра, которым присваиваются некоторые значения свойств класса. Запусти программу и внимательно посмотри на её вывод. Что не так? Почему программа выводит не те данные?
Исправь ошибки!'''
''' Перед тобой программа, в которой описан класс «Прямоугольник». В нём есть четыре свойства: координата угла x, координата угла y, ширина и высота прямоугольника.
В программе создано два экземпляра, которым присваиваются некоторые значения свойств класса. Запусти программу и внимательно посмотри на её вывод. Что не так? Почему программа выводит не те данные?
Исправь ошибки!'''
'''
class Rectangle():
  
  def setproperty(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height

figure1 = Rectangle()
figure1.setproperty(5, 5, 10, 5)
figure2 = Rectangle()
figure2.setproperty(7 ,6 ,3 ,8 )

print("Координата x: " + str(figure1.x) + ", координата y: " + str(figure1.y) + ", ширина: " + str(figure1.width) + ", высота: " + str(figure1.height))
print("Координата x: " + str(figure2.x) + ", координата y: " + str(figure2.y) + ", ширина: " + str(figure2.width) + ", высота: " + str(figure2.height))
'''

'''
class Rectangle():
  self.x = 0
  self.y = 0
  self.width = 0
  self.height = 0

figure1 = Rectangle()
figure1.x = 5
figure1.y = 5
figure1.width = 10
figure1.height = 5

figure2 = Rectangle()
figure2.x = 7
figure2.y = 6
figure2.widtsh = 3
figure2.heiths = 8

print("Координата x: " + str(figure1.x) + ", координата y: " + str(figure1.y) + ", ширина: " + str(figure1.width) + ", высота: " + str(figure1.height))
print("Координата x: " + str(figure2.x) + ", координата y: " + str(figure2.y) + ", ширина: " + str(figure2.width) + ", высота: " + str(figure2.height))
'''

'''
Перед тобой программа, в которой описан класс «Квадрат». В нём определён метод create(), с помощью которого экземплярам задаются значения свойств.
Исправь все ошибки. Результатом программы должен быть запрос у пользователя значений трёх свойств для каждого из экземпляров и их вывод, как показано в примере.'''
'''
class Square():
    def create(self, x, y, side_length):
        self.x = x
        self.y = y
        self.side_length = side_length

figure1 = Square()
x = int(input("Введите координату x: "))
y = int(input("Введите координату y: "))
side_length = int(input("Введите длину стороны квадрата: "))
figure1.create(x, y, side_length)
print(str(figure1.x) + " " + str(figure1.y) + " " + str(figure1.side_length))

figure2 = Square()
x = int(input("Введите координату x: "))
y = int(input("Введите координату y: "))
side_length = int(input("Введите длину стороны квадрата: "))
figure2(x, y, side_length)
print(str(figure2.x) + " " + str(figure2.y) + " " + str(figure2.side_length))
'''


''' Класс «Круг»
Перед тобой программа, в которой описан класс «Круг» и определён метод inside().
Допиши программу, последовательно выполняя шаги, записанные ниже.
1) Создай экземпляр класса.
Программа должна запрашивать у пользователя координаты центра окружности (x1, y1) и длину радиуса.
2) Ввод координат трёх различных точек.
Программа должна три раза запрашивать ввод координат точки (x2, y2) и, если расстояние от этой точки до центра окружности меньше радиуса — выводить True, иначе — False. Все необходимые для этого формулы и условия уже записаны в методе inside(). Формат ввода/вывода смотри на картинке.'''

import math

class Circle():
  
    def inside(self, x1, y1, x2, y2, radius):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.radius = radius
        self.length = math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        if self.length < self.radius:
            return True
        else:
            return False


example1 = Circle()
a = int(input('Введите X центра: '))
b = int(input('Введите Y центра: '))
R = int(input('Введите Radius: '))
n = 0
while n != 3:
  c = int(input('Введите координату X: '))
  d = int(input('Введите координату Y: '))
  print(example1.inside(a, b, c, d, R))
  n += 1