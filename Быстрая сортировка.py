import random


def quicksort(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = random.choice(nums)
    l_nums = [n for n in nums if n < q]
    e_nums = [q] * nums.count(q)
    b_nums = [n for n in nums if n > q]
    return quicksort(l_nums) + e_nums + quicksort(b_nums)


N = int(input())
list_of_3 = []
list_of_nums = list([int(x) if (int(x) % 3 != 0) else list_of_3.append(int(x)) for x in input().split()])
list_of_nums = list(filter(None, list_of_nums))
sorted_list = quicksort(list_of_3) + quicksort(list_of_nums)
print(sorted_list)

# 5 2 6 8 1 9 3 7 2
# 1 2 3 4 5 6 7 8 9