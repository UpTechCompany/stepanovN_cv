import numpy as np
import matplotlib.pyplot as plt
import socket

# Новые значения IP-адреса и порта
server_ip = "84.237.21.36"
server_port = 5152

# Функция для определения соседей точки
def get_neighbors(y, x):
    return [(y - 1, x), (y - 1, x - 1), (y - 1, x + 1), (y + 1, x + 1), (y + 1, x), (y, x - 1), (y, x + 1),  (y + 1, x - 1)]

# Проверка соседних точек
def check_neighbors(arr, y, x):
    for neighbor in get_neighbors(y, x):
        if arr[y][x] <= arr[neighbor[0]][neighbor[1]]:
            return False
    return True

# Получение данных от сервера
def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return
        data.extend(packet)
    return data

# Установка соединения с сервером
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
    connection.connect((server_ip, server_port))
    for _ in range(10):
        connection.send(b"get")
        received_bytes = receive_all(connection, 40002)

        image = np.frombuffer(received_bytes[2:40002], dtype="uint8").reshape(received_bytes[0], received_bytes[1])

        max_coords = []
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if check_neighbors(image, y, x):
                    max_coords.append((y, x))
        result = np.sqrt((max_coords[0][0] - max_coords[1][0]) ** 2 + (max_coords[0][1] - max_coords[1][1]) ** 2)
        connection.send(f"{round(result, 1)}".encode())
        print(connection.recv(20))
    connection.send(b"beat")
    beat_msg = connection.recv(20)
    print(beat_msg)
