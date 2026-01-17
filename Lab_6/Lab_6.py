import cv2
import os
import time
import numpy as np

# Налаштування шляхів до ресурсів
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, 'assets')

path_cars = os.path.join(assets_dir, 'cars.xml')
path_people = os.path.join(assets_dir, 'haarcascade_fullbody.xml')
path_moto = os.path.join(assets_dir, 'two_wheeler.xml')

video_source = 'traffic.mp4'

# Ініціалізація каскадних класифікаторів
car_cascade = cv2.CascadeClassifier(path_cars)
people_cascade = cv2.CascadeClassifier(path_people)
moto_cascade = cv2.CascadeClassifier(path_moto)

# Ініціалізація алгоритму віднімання фону MOG2
# history=500: кількість кадрів для побудови моделі фону
# varThreshold=25: поріг чутливості (Mahalanobis distance)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

cap = cv2.VideoCapture(video_source)

print("Розпочато обробку відео...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Зміна розміру кадру для підвищення продуктивності
    height, width = frame.shape[:2]
    new_width = 640
    ratio = new_width / width
    frame = cv2.resize(frame, (new_width, int(height * ratio)))

    # Обробка маски руху
    fgMask = backSub.apply(frame)

    # Фільтрація шумів на масці (порогове перетворення та морфологічні операції)
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    fgMask = cv2.erode(fgMask, None, iterations=1)
    fgMask = cv2.dilate(fgMask, None, iterations=5)

    # Підготовка зображення для класифікаторів
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекція об'єктів
    cars = car_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    people = people_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    motos = moto_cascade.detectMultiScale(gray, 1.05, 2, minSize=(40, 40))

    # Функція перевірки динаміки об'єкта
    # Відсіює статичні помилкові детекції (наприклад, дерева)
    def is_moving(x, y, w, h, mask, threshold=0.03):
        roi = mask[y:y + h, x:x + w]
        white_pixels = cv2.countNonZero(roi)
        total_pixels = w * h
        if total_pixels == 0: return False
        movement_ratio = white_pixels / total_pixels
        return movement_ratio > threshold

    # Візуалізація автомобілів
    for (x, y, w, h) in cars:
        if is_moving(x, y, w, h, fgMask):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Car', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Візуалізація людей
    for (x, y, w, h) in people:
        if is_moving(x, y, w, h, fgMask):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Візуалізація мотоциклів з перевіркою накладання
    for (x, y, w, h) in motos:
        if is_moving(x, y, w, h, fgMask):
            inside_car = False
            # Перевірка, чи не знаходиться мотоцикл всередині рамки автомобіля
            for (cx, cy, cw, ch) in cars:
                mx, my = x + w // 2, y + h // 2
                if cx < mx < cx + cw and cy < my < cy + ch:
                    inside_car = True
                    break

            if not inside_car:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(frame, 'Moto', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Вивід результатів
    cv2.imshow('Motion Mask', cv2.resize(fgMask, (400, 300)))
    cv2.imshow('Lab 6: Hierarchical Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()