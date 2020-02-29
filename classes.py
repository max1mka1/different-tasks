class Square():
    def create(x, y, side_length):
        self.x = x
        self.y = y
        side_length = side_length

figure1 = Square()
x = int(input("Введите координату x: "))
y = int(input("Введите координату y: "))
side_length = int(input("Введите длину стороны квадрата: "))
figure1.create(x, y, side_length)
print(str(figure1.x) + " " + str(figure1.y) + " " + str(figure1.side_length))

figure2 = Square()
x = int(input("Введите координату x: "))
y = int(input("Введите координату y: "))
side_length = int(input("Введите длину стороны квадрата: "))
figure2(x, y, side_length)
print(str(figure2.x) + " " + str(figure2.y) + " " + str(figure2.side_length))