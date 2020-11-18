import re

text = r'(2(3AC)A)'

subgroup_pattern = r"[(]\d+\w+[)]"
text_pattern = r"\D+"
count_pattern = r"\d+"
result = re.finditer(subgroup_pattern, text)
while '(' in text:
  for x in result:
    a, b, subgroup = x.start(), x.end(), x.group()
  #print(a, b, subgroup)
  subtext = text[a+1:b-1]
  count = int(re.search(count_pattern, subtext).group(0))
  sub_text = re.search(text_pattern, subtext).group(0)
  #print(count, sub_text)
  replace_text = sub_text * count
  #print(f"replace_text = {replace_text}, {type(replace_text)}")
  #print(f"subgroup = {subgroup}, {type(subgroup)}")
  #print(f"text = {text}, {type(text)}")
  text = text.replace(subgroup, replace_text)
  #print(f"text = {text}")
  result = re.finditer(subgroup_pattern, text)
print(text)