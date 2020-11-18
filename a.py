n = int(input())

if (n - 3) % 3 == 0:
	a = (n - 3) // 3
	print(a, a + 1, a + 2)
	
else:
	print(-1)