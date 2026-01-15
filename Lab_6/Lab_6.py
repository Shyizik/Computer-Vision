import cv2
import os

# --- 1. НАЛАШТУВАННЯ ШЛЯХІВ ---
# Визначаємо шлях до папки assets відносно скрипта
base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, 'assets')

path_cars = os.path.join(assets_dir, 'cars.xml')
path_people = os.path.join(assets_dir, 'haarcascade_fullbody.xml')
path_dog = os.path.join(assets_dir, 'dog_face.xml')

# Відеофайл (має бути поруч зі скриптом або вкажіть повний шлях)
video_source = 'traffic.mp4'

# --- 2. ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ---
print(">>> Завантаження каскадів...")

car_cascade = cv2.CascadeClassifier(path_cars)
people_cascade = cv2.CascadeClassifier(path_people)
dog_cascade = cv2.CascadeClassifier(path_dog)

# Перевірка на помилки завантаження
if car_cascade.empty(): print(f"⚠️ Помилка: Не знайдено {path_cars}")
if people_cascade.empty(): print(f"⚠️ Помилка: Не знайдено {path_people}")
if dog_cascade.empty(): print(f"⚠️ Помилка: Не знайдено {path_dog}")

# --- 3. ЗАПУСК ВІДЕО ---
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Помилка: Не вдалося відкрити відео {video_source}")
    exit()

print("✅ Система готова. Натисніть 'q' для виходу.")

while True:
    # Читання кадру
    ret, frame = cap.read()
    if not ret:
        break  # Кінець відео або помилка читання

    # --- 4. ОБРОБКА ЗОБРАЖЕННЯ (Pre-processing) ---
    # Зменшення розміру для підвищення FPS
    frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

    # Перетворення в сірий (вимога алгоритму Хаара)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Розмиття для видалення шумів (зменшує хибні спрацювання)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 5. ДЕТЕКЦІЯ ОБ'ЄКТІВ ---

    # 5.1 Пошук машин (minNeighbors=3)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    # 5.2 Пошук людей
    people = people_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # 5.3 Пошук собак (minNeighbors=5 для більшої точності, бо модель "слабка")
    dogs = dog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # --- 6. ВІЗУАЛІЗАЦІЯ ---

    # Малюємо машини (СИНІЙ)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Малюємо людей (ЗЕЛЕНИЙ)
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Малюємо собак (ЖОВТИЙ)
    for (x, y, w, h) in dogs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, 'Dog', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- 7. СТАТИСТИКА ---
    info_text = f"Cars: {len(cars)} | People: {len(people)} | Dogs: {len(dogs)}"
    # Чорна підкладка для тексту
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Показ вікна
    cv2.imshow('Lab 6: Multi-Object Detection', frame)

    # Вихід на клавішу 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# --- 8. ОЧИЩЕННЯ РЕСУРСІВ ---
cap.release()
cv2.destroyAllWindows()