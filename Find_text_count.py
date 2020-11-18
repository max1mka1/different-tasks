import re

text = r'(3AC)'

# Так можно найти число повторов из строки
pattern = r"\d+"
result_count = re.findall(pattern, text)
count = int(result_count[0])
print(count)

# А так можно найти саму подстроку, которую нужно продублировать
pattern = r"\w+"
result_subtext = re.findall(pattern, text)
subtext = result_subtext[0]
print(subtext)