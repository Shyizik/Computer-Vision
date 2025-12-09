import cv2
import numpy as np
import os

# ==========================================
# НАЛАШТУВАННЯ ФАЙЛІВ
# ==========================================
FILE_LANDSAT = 'low.jpg'
FILE_BING = 'bing.jpg'
# ==========================================

# Глобальні змінні
img_display = None
mode_current = None
window_name = None
trackbars_ready = False


def on_trackbar(val):
    if not trackbars_ready: return
    if img_display is None: return

    output = img_display.copy()
    h_img, w_img = img_display.shape[:2]

    # =======
    # LANDSAT
    # =======
    if mode_current == 'low':
        blur_val = cv2.getTrackbarPos('Blur', window_name)
        thresh_val = cv2.getTrackbarPos('Threshold', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)
        contrast_val = cv2.getTrackbarPos('Contrast', window_name)

        if blur_val % 2 == 0: blur_val += 1
        if blur_val < 1: blur_val = 1
        clip_limit = contrast_val / 10.0 if contrast_val > 0 else 0.1

        # 1. Покращення зображення
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced, blur_val)

        # Інверсія
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Морфологія
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                # Фільтрація країв
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5:
                    continue

                # РАМКИ
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 20)

                # НОМЕРИ
                label = str(count + 1)

                # Чорна обводка
                cv2.putText(output, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 0), 30)
                # Колір всередині
                cv2.putText(output, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 255, 255), 10)
                count += 1

        # ГОЛОВНИЙ ЛІЧИЛЬНИК
        text_counter = f"FOUND: {count}"

        # Чорна обводка лічильника
        cv2.putText(output, text_counter, (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 10.0, (0, 0, 0), 40)

        # Білий текст лічильника
        cv2.putText(output, text_counter, (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 10.0, (255, 255, 255), 15)

        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', binary)

    # ====
    # BING
    # ====
    elif mode_current == 'bing':
        c_min = cv2.getTrackbarPos('Canny Min', window_name)
        c_max = cv2.getTrackbarPos('Canny Max', window_name)
        thick = cv2.getTrackbarPos('Thickness', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)
        rect_limit = cv2.getTrackbarPos('Rectangularity', window_name) / 100.0

        if thick < 1: thick = 1

        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, c_min, c_max)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.dilate(edges, kernel, iterations=thick)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5: continue

                rect_area = w * h
                extent = float(area) / rect_area
                if extent < rect_limit: continue

                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Стандартні рамки
                cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)

                # Стандартні номери
                label = str(count + 1)
                cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                count += 1

        # Стандартний лічильник
        cv2.putText(output, f"Buildings Found: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', closed)


def start_processing(mode):
    # Оголошуємо глобальні змінні, щоб мати до них доступ у функції on_trackbar
    global img_display, mode_current, window_name, trackbars_ready

    trackbars_ready = False
    mode_current = mode

    # Вибираємо файл зображення залежно від режиму:
    # 'low' - для знімка Landsat (низька якість)
    # інакше - для знімка Bing (висока якість)
    filename = FILE_LANDSAT if mode == 'low' else FILE_BING

    # Перевірка: чи існує файл за вказаним шляхом
    if not os.path.exists(filename):
        print(f"ПОМИЛКА: Не знайдено файл {filename}")
        return

    # Завантаження зображення у пам'ять
    original = cv2.imread(filename)
    if original is None:
        print(f"ПОМИЛКА: Не вдалося відкрити {filename}")
        return

    # Отримуємо початкові розміри зображення
    target_width = 1200
    h, w = original.shape[:2]

    # --- БЛОК МАСШТАБУВАННЯ (RESIZING) ---
    if mode == 'low':
        # Для Landsat: Збільшуємо зображення в 10 разів, щоб краще бачити пікселі/плями
        scale = 10
        # Використовуємо кубічну інтерполяцію (INTER_CUBIC) для якісного збільшення
        img_display = cv2.resize(original, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        window_name = "Landsat Settings"
    else:
        # Для Bing: Зменшуємо зображення, якщо воно ширше за target_width (1200px),
        # щоб вікно влазило в екран монітора
        scale = target_width / w if w > target_width else 1
        img_display = cv2.resize(original, (int(w * scale), int(h * scale))) if scale != 1 else original.copy()
        window_name = "Bing Settings"

    # --- НАЛАШТУВАННЯ ВІКОН ---
    # Створюємо головне вікно з можливістю зміни розміру
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    # Створюємо допоміжне вікно для перегляду маски (чорно-біле)
    cv2.namedWindow('Mask View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask View', 500, 350)

    # --- СТВОРЕННЯ ПОВЗУНКІВ (TRACKBARS) ---
    if mode == 'low':
        # Повзунки для режиму Landsat:
        # Blur - сила розмиття (для згладжування шуму)
        # Threshold - поріг бінаризації (відсіювання фону)
        # Min Area - фільтр дрібних об'єктів за площею
        # Contrast - налаштування CLAHE (покращення видимості)
        cv2.createTrackbar('Blur', window_name, 3, 30, on_trackbar)
        cv2.createTrackbar('Threshold', window_name, 100, 255, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 500, 5000, on_trackbar)
        cv2.createTrackbar('Contrast', window_name, 40, 100, on_trackbar)
    else:
        # Повзунки для режиму Bing:
        # Canny Min/Max - пороги для детектора меж Кенні
        # Thickness - товщина ліній при об'єднанні контурів
        # Rectangularity - фільтр "прямокутності" (відсіює дерева)
        cv2.createTrackbar('Canny Min', window_name, 50, 255, on_trackbar)
        cv2.createTrackbar('Canny Max', window_name, 150, 255, on_trackbar)
        cv2.createTrackbar('Thickness', window_name, 2, 10, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 300, 5000, on_trackbar)
        cv2.createTrackbar('Rectangularity', window_name, 35, 100, on_trackbar)

    # Прапорець готовий, дозволяємо оновлення картинки
    trackbars_ready = True

    # Викликаємо функцію обробки вручну перший раз, щоб показати картинку одразу
    on_trackbar(0)
    print(f"Вікно відкрито. Натисніть 'q' для виходу.")

    # --- ГОЛОВНИЙ ЦИКЛ ---
    while True:
        # Чекаємо натискання клавіші 100 мс
        # Якщо натиснуто 'q' (код клавіші), виходимо з циклу
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Закриваємо всі вікна OpenCV перед виходом
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Запит користувача на вибір режиму роботи
    choice = input("\nВведіть 1 (Landsat) або 2 (Bing): ")
    if choice == '1':
        start_processing('low')
    elif choice == '2':
        start_processing('bing')