# Project 1
# @ Linear equation solver
# Input: Linear equation
# 8 + 3 * x = 27
# Output: solution
# x = 19 / 3 = 6.(3)
import re

def disintegrator(subequation): # Добавить pattern для known, что бы вычленить из subequation числа со знаком перед ними. pattern - правило.
    '''Function that finds all known and unknown elements of subequation'''
    pattern_x = re.compile(r'[+-]?\d+x|[+-]?\d+\*x')  # \d - any digit
    unknown = re.findall(pattern_x, subequation)
    subequation_ = re.sub(r"[+-]?\d+x|[+-]?\d+\*x", '', subequation)
    pattern_numbers = re.compile(r'[+-]?\d+')
    known = re.findall(pattern_numbers, subequation_)
    return known, unknown

def summ(known): 
    pass


equation = "-98 + 2X - 843 + 21 - 0x - 7 * x+8=-39*x+966 + 8 -10".lower() # input("Input your equation:").lower()
chek_equal = re.findall(r'=', equation)
quantity_of_equals = len(chek_equal)
if quantity_of_equals != 1:
    print(f'The equation is incorrect: quantity of equals = {quantity_of_equals} !')
else:
    equation = re.sub(r' ', '', equation)
    equation = re.split(r'=', equation)
    subequation_left = equation[0]
    subequation_right = equation[1]
    print(f'subequation_left = {subequation_left}')
    known_left, unknown_left = disintegrator(subequation_left)
    print(f'known_left = {known_left}')
    print(f'unknown_left = {unknown_left}')
    known_left_sum = summ(known_left)
    known_right, unknown_right = disintegrator(subequation_right)
    print(f'subequation_right = {subequation_right}')
    print(f'known_right = {known_right}')
    print(f'unknown_right = {unknown_right}')
