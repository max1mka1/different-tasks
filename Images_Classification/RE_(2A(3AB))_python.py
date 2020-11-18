import re

text = r'(2(3AC)A)'
#text = r'(15a)B'

subgroup_pattern = r"[(]\d+\w+[)]"
text_pattern = r"\D+"
count_pattern = r"\d+"
result = re.finditer(subgroup_pattern, text)
while '(' in text:
  for x in result:
    a, b, subgroup = x.start(), x.end(), x.group()
    print(a, b, subgroup)