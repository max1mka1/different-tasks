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
  subtext = text[a+1:b-1]
  count = int(re.search(count_pattern, subtext).group(0))
  sub_text = re.search(text_pattern, subtext).group(0)
  replace_text = sub_text * count
  text = text.replace(subgroup, replace_text)
  result = re.finditer(subgroup_pattern, text)
print(text)