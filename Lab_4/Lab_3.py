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

    # ==========================================================
    # РЕЖИМ 1: LANDSAT (Виправлено: Лічильник зліва зверху, пошук всюди)
    # ==========================================================
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

        # Інверсія для чорних будинків
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

                # Фільтрація країв картинки (щоб не ловити рамку самого фото)
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5:
                    continue

                # Малюємо прямокутник навколо будинку
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Номер будинку біля самого об'єкта
                cv2.putText(output, str(count + 1), (cx - 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                count += 1

        # === ЛІЧИЛЬНИК (У лівому верхньому куті) ===
        # Малюємо темну підкладку під текст, щоб було видно (якщо фон світлий)
        cv2.rectangle(output, (0, 0), (450, 60), (0, 0, 0), -1)  # Чорний прямокутник

        # Сам текст лічильника
        cv2.putText(output, f"Buildings (Landsat): {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', binary)

    # ==========================================================
    # РЕЖИМ 2: BING (Без змін)
    # ==========================================================
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

                cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
                cv2.putText(output, str(count + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                count += 1

        # Лічильник Bing
        cv2.putText(output, f"Buildings Found: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', closed)


def start_processing(mode):
    global img_display, mode_current, window_name, trackbars_ready

    trackbars_ready = False
    mode_current = mode
    filename = FILE_LANDSAT if mode == 'low' else FILE_BING

    if not os.path.exists(filename):
        print(f"ПОМИЛКА: Не знайдено файл {filename}")
        return

    original = cv2.imread(filename)
    if original is None:
        print(f"ПОМИЛКА: Не вдалося відкрити {filename}")
        return

    target_width = 1200
    h, w = original.shape[:2]

    if mode == 'low':
        scale = 10
        img_display = cv2.resize(original, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        window_name = "Landsat Settings (Corrected)"
    else:
        scale = target_width / w if w > target_width else 1
        img_display = cv2.resize(original, (int(w * scale), int(h * scale))) if scale != 1 else original.copy()
        window_name = "Bing Settings"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)
    cv2.namedWindow('Mask View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask View', 500, 350)

    if mode == 'low':
        cv2.createTrackbar('Blur', window_name, 3, 30, on_trackbar)
        cv2.createTrackbar('Threshold', window_name, 100, 255, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 500, 5000, on_trackbar)
        cv2.createTrackbar('Contrast', window_name, 40, 100, on_trackbar)
    else:
        cv2.createTrackbar('Canny Min', window_name, 50, 255, on_trackbar)
        cv2.createTrackbar('Canny Max', window_name, 150, 255, on_trackbar)
        cv2.createTrackbar('Thickness', window_name, 2, 10, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 300, 5000, on_trackbar)
        cv2.createTrackbar('Rectangularity', window_name, 35, 100, on_trackbar)

    trackbars_ready = True
    on_trackbar(0)
    print(f"Вікно відкрито. Натисніть 'q' для виходу.")

    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("\nВведіть 1 (Landsat) або 2 (Bing): ")
    if choice == '1':
        start_processing('low')
    elif choice == '2':
        start_processing('bing')