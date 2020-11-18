import serial

baud_rate = 4800 #whatever baudrate you are listening to
com_port1 = 'COM6' #replace with your first com port path
com_port2 = 'COM5' #replace with your second com port path

listener = serial.Serial(com_port1, baud_rate)
forwarder = serial.Serial(com_port2, baud_rate)

while 1:
    serial_out = listener.read(size=1)
    print(serial_out) #or write it to a file 
    forwarder.write(4)