
print('Stupit list creation')
a = []
for i in range(1, 11):
    a.append(i)

print(a, end='\n\n')

print('Generator usage')
a = [i for i in range(1, 11)]
print(a, end='\n\n')

a = [i for i in range(30, 41)]
print(a)


print('Generator usage with a function')
def square(x):
    return x ** 2

print(square(6))

a = [square(i) for i in range(1, 11)]
print(a, end='\n\n')

# функция прибавляет 6
# используй её в генераторе (-6 до 15)

def kv(x):
  return x + 6
a = [kv(i) for i in range(-6, 15)]
print(a)



# Generator usage with a lamda function
f = lambda x: x ** 2
a = [f(i) for i in range(1, 11)]
print(a, end='\n\n')



f = lambda x: x + 6
a = [f(i) for i in range(-6, 15)]
print(a)

# Generator usage with a lamda function with condition
f = lambda x: x ** 2
a = [f(i) for i in range(1, 11) if i % 2 == 0]
# i % 2 - остаток от деления на 2
# Возведи в квадрат числа, кратные 2м, на промежутке от 1 до 10 ВКЛЮЧИТЕЛЬНО
print(a, end='\n\n')

# print(6 % 2) # остаток
# print(6 // 2) # целочисленное деление

# ВПЕРЕД ВЛЕВО РАЗВЕРНИСЬ ВПЕРЕД ВПЕРЕД


# Generator usage with a lamda function with condition
print('Generator usage with words:')

word_to_lower = lambda x: x.lower()
a = [word_to_lower(word) for word in ['PUKA', 'MUKA', 'POL'] if word[0] == 'P']
#  Приведи к нижнему регистру только те слова, что начинаются на букву P
print(a, end='\n\n')

# print('HeLoeErdDFg'.lower())

# upper()
word_to_lower = lambda x: x.upper()
a = [word_to_lower(word) for word in ['puka', 'kororovirusnaya muka', 'pol'] if word[0] == 'k']
print(a, end='\n\n')


print('Matrix numbers generation 2D')
# b = [i for i in range(1, 11)]
a = [[i for i in range(1, 11)] for j in range(1, 15)]
print(a, end='\n\n')

# Задача 1. Сгенерировать таблицу умножения от 0 до 10.

f = lambda x, y: x * y
a = [[f(i, j) for i in range(1, 11)] for j in range(1, 11)]
print(a)

# Задача 2. Сгенерировать список случайных чисел до 20
# import random

# print(random.randint(1, 10))




a = [randint(1, 11) for i in range (1, 11)]
print(a)


# Задача 3. Сгенерировать список из 10 любых чисел от 100 до 500. Сгенерировать список чисел, на основании этого списка, так, чтобы если в числе есть цифра 1, вывести квадрат числа, иначе поделить на 10
from random import randint
kvadrat = lambda x: x ** 2
print('1' in str(100))
a = [randint(100, 500) for i in range (1, 11)]
f =  [kvadrat(number) if '1' in str(number) else number / 100 for number in a]
print(f)