import cv2
import os
import time
import numpy as np

# --- 1. НАЛАШТУВАННЯ ---
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, 'assets')

path_cars = os.path.join(assets_dir, 'cars.xml')
path_people = os.path.join(assets_dir, 'haarcascade_fullbody.xml')
path_moto = os.path.join(assets_dir, 'two_wheeler.xml')  # Спробуйте знайти кращий xml, якщо цей не спрацює

video_source = 'traffic.mp4'

# --- 2. ІНІЦІАЛІЗАЦІЯ ---
car_cascade = cv2.CascadeClassifier(path_cars)
people_cascade = cv2.CascadeClassifier(path_people)
moto_cascade = cv2.CascadeClassifier(path_moto)

# ІНІЦІАЛІЗАЦІЯ ДЕТЕКТОРА РУХУ (Це і є фільтр динаміки)
# history=500: пам'ятає останні 500 кадрів для фону
# varThreshold=16: чутливість (менше -> чутливіше)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

cap = cv2.VideoCapture(video_source)

print("🚀 Старт. Використовуємо ієрархію: Рух -> Каскади")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Оптимізація розміру
    height, width = frame.shape[:2]
    new_width = 640
    ratio = new_width / width
    frame = cv2.resize(frame, (new_width, int(height * ratio)))

    # --- ЕТАП 1: ВИЯВЛЕННЯ РУХУ (ДИНАМІКИ) ---
    # Створюємо маску руху: біле = рухається, чорне = фон
    fgMask = backSub.apply(frame)

    # Прибираємо шуми (тіні, дрібні відблиски)
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    fgMask = cv2.erode(fgMask, None, iterations=1)
    fgMask = cv2.dilate(fgMask, None, iterations=2)

    # Підготовка для каскадів
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- ЕТАП 2: КЛАСИФІКАЦІЯ ---

    # Знижуємо minNeighbors, щоб зловити мотоцикл,
    # але фільтруємо через маску руху, щоб прибрати помилки
    cars = car_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    # Для людей ставимо дуже низький поріг, бо дерево відсіється рухом
    people = people_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    motos = moto_cascade.detectMultiScale(gray, 1.05, 2, minSize=(40, 40))  # 2 сусіда - дуже агресивний пошук


    # Функція для перевірки: чи об'єкт рухається?
    def is_moving(x, y, w, h, mask, threshold=0.15):
        # Вирізаємо шматок маски руху під об'єктом
        roi = mask[y:y + h, x:x + w]
        # Рахуємо відсоток білих пікселів (руху)
        white_pixels = cv2.countNonZero(roi)
        total_pixels = w * h
        if total_pixels == 0: return False
        movement_ratio = white_pixels / total_pixels
        return movement_ratio > threshold  # Якщо більше 15% площі рухається - це об'єкт


    # --- МАЛЮВАННЯ З ПЕРЕВІРКОЮ РУХУ ---

    # Машини
    for (x, y, w, h) in cars:
        if is_moving(x, y, w, h, fgMask):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Car', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Люди
    for (x, y, w, h) in people:
        # Дерево не пройде цю перевірку!
        if is_moving(x, y, w, h, fgMask):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Мотоцикли
    for (x, y, w, h) in motos:
        if is_moving(x, y, w, h, fgMask):
            # Додаткова перевірка, щоб не малювати мотоцикл всередині машини (як у вашому коді)
            inside_car = False
            for (cx, cy, cw, ch) in cars:
                mx, my = x + w // 2, y + h // 2
                if cx < mx < cx + cw and cy < my < cy + ch:
                    inside_car = True
                    break

            if not inside_car:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(frame, 'Moto', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Показуємо результат і маску (для налагодження)
    cv2.imshow('Motion Mask', cv2.resize(fgMask, (400, 300)))  # Маленьке вікно, щоб бачити, що рухається
    cv2.imshow('Lab 6: Hierarchical Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()