class Task():
    description = ''
    importance = 1
    completion_month = ''
    completion_date = 1

def creation():
    objective = Task()
    objective.description = input("Введите описание задачи: ")
    objective.importance = int(input("Введите важность задачи (от 1 до 5): "))
    objective.completion_month = input("Введите месяц сдачи: ")
    objective.completion_date = int(input("Введите дату сдачи: "))
    return(objective)


list_of_tasks = []
m = int(input("Please, enter the number of tasks"))
i = 0
while i != m:
  task = creation()
  list_of_tasks.append(task)
  i = i + 1

i = 0
max_importance = 0
max_urgency = 100
max_importance_task = ''
max_urgency_task = ''
for task in list_of_tasks:
  if task.importance > max_importance:
    max_importance = task.importance
    max_importance_task = task
  if task.completion_date < max_urgency:
    max_urgency= task.completion_date
    max_urgency_task = task
  i = i + 1
i = max_importance_task
print("Самая важная задача: " + i.description + ", важность- " + str(i.importance) + ", месяц сдачи- " + i.completion_month + ", дата сдачи- " + str(i.completion_date))
i = max_urgency_task
print("Самая срочная задача: " + i.description + ", важность- " + str(i.importance) + ", месяц сдачи- " + i.completion_month + ", дата сдачи- " + str(i.completion_date))