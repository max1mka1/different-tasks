a = int(input())
if a == 1000:
	print(1)
else:
	if a < 1000:
		if a % 3 == 0:
			print(2)
		else:
			print(1)
	else:
		a = a % 1000
		if a % 3 == 0:
			print(1)
		else:
			print(2)
