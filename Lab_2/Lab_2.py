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

    # === LANDSAT (Низька якість - Плями) ===
    if mode_current == 'low':
        blur_val = cv2.getTrackbarPos('Blur', window_name)
        thresh_val = cv2.getTrackbarPos('Threshold', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)

        if blur_val % 2 == 0: blur_val += 1
        if blur_val < 1: blur_val = 1

        # Фільтрація шумів
        blurred = cv2.medianBlur(img_display, blur_val)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Сегментація (бінаризація)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Фільтр рамки
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5: continue

                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, str(count + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                count += 1

        cv2.putText(output, f"Objects: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', binary)

    # === BING (Висока якість - Контури/Стіни) ===
    elif mode_current == 'bing':
        c_min = cv2.getTrackbarPos('Canny Min', window_name)
        c_max = cv2.getTrackbarPos('Canny Max', window_name)
        thick = cv2.getTrackbarPos('Thickness', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)

        # Геометричний фільтр форми (відсіює дерева)
        rect_limit = cv2.getTrackbarPos('Rectangularity', window_name) / 100.0

        if thick < 1: thick = 1

        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)

        # Розмиття: прибирає текстуру трави, залишаючи чіткі межі стін
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # 1. Детекція меж (Canny Edge Detection) - знаходить стіни та перепади
        edges = cv2.Canny(blurred, c_min, c_max)

        # 2. Морфологія (Dilation) - з'єднує розірвані лінії стін у суцільний контур
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.dilate(edges, kernel, iterations=thick)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Векторизація - пошук замкнених контурів будівель
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)

                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5: continue

                # === ГЕОМЕТРИЧНИЙ АНАЛІЗ ===
                rect_area = w * h
                extent = float(area) / rect_area

                # Фільтр: Дерева мають "рвані" краї (низький extent), Будівлі - рівні (високий)
                if extent < rect_limit:
                    continue

                # Апроксимація: спрощення форми до геометричного примітиву (полігону)
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
                cv2.putText(output, str(count + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                count += 1

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

    # Масштабування
    target_width = 1200
    h, w = original.shape[:2]

    if mode == 'low':
        scale = 10
        img_display = cv2.resize(original, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        window_name = "Landsat Settings"
    else:
        scale = target_width / w if w > target_width else 1
        img_display = cv2.resize(original, (int(w * scale), int(h * scale))) if scale != 1 else original.copy()
        window_name = "Bing Settings"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    cv2.namedWindow('Mask View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask View', 500, 350)

    if mode == 'low':
        cv2.createTrackbar('Blur', window_name, 15, 30, on_trackbar)
        cv2.createTrackbar('Threshold', window_name, 100, 255, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 500, 5000, on_trackbar)
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
