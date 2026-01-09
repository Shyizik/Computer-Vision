import cv2
import numpy as np
import os

# ==========================================
# КОНФІГУРАЦІЯ
# ==========================================
FILE_LANDSAT = 'low.jpg'
FILE_BING = 'bing.jpg'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
# ==========================================

img_display = None
mode_current = None
window_name = None
trackbars_ready = False


def on_trackbar(val):
    if not trackbars_ready or img_display is None: return

    output = img_display.copy()
    h_img, w_img = img_display.shape[:2]

    # Змінна для відображення маски
    display_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    # ==========================================================
    # РЕЖИМ 1: LANDSAT (ЕКСТРЕМАЛЬНА ВИДИМІСТЬ)
    # ==========================================================
    if mode_current == 'low':
        thresh_white = cv2.getTrackbarPos('White Thresh', window_name)
        thresh_black = cv2.getTrackbarPos('Black Thresh', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)

        hsv = cv2.cvtColor(img_display, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced, 7)

        _, mask_white = cv2.threshold(blurred, thresh_white, 255, cv2.THRESH_BINARY)
        _, mask_dark = cv2.threshold(blurred, thresh_black, 255, cv2.THRESH_BINARY_INV)
        mask_buildings = cv2.bitwise_or(mask_white, mask_dark)

        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_grass = cv2.inRange(hsv, lower_green, upper_green)

        lower_yellow = np.array([10, 10, 140])
        upper_yellow = np.array([40, 120, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_garbage = cv2.bitwise_or(mask_grass, mask_yellow)
        mask_keep = cv2.bitwise_not(mask_garbage)

        final_mask = cv2.bitwise_and(mask_buildings, mask_buildings, mask=mask_keep)

        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        display_mask = final_mask

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 2 or y <= 2 or (x + w) >= w_img - 2 or (y + h) >= h_img - 2: continue

                # === ЗМІНИ ТУТ: ДУЖЕ ТОВСТІ ЛІНІЇ ===
                # Рамка товщиною 5 (щоб було видно на пікселях)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # Підкладка під текст (щоб цифри не зливалися з фоном)
                cv2.rectangle(output, (x, y - 50), (x + 70, y), (0, 0, 0), -1)

                # Величезний текст (розмір 2.0, товщина 4)
                cv2.putText(output, str(count + 1), (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                count += 1

        # Лічильник теж великий
        cv2.rectangle(output, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.putText(output, f"Found: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # ==========================================================
    # РЕЖИМ 2: BING
    # ==========================================================
    elif mode_current == 'bing':
        blur_val = cv2.getTrackbarPos('Texture Smooth', window_name)
        contrast_clip = cv2.getTrackbarPos('Contrast Boost', window_name)
        dark_thresh = cv2.getTrackbarPos('Dark Thresh', window_name)
        max_sat = cv2.getTrackbarPos('Max Saturation', window_name)
        solidity_thresh = cv2.getTrackbarPos('Solidity %', window_name) / 100.0
        remove_roads = cv2.getTrackbarPos('Road Filter', window_name) / 100.0
        min_area = cv2.getTrackbarPos('Min Area', window_name)

        if blur_val < 1: blur_val = 1

        smoothed = cv2.bilateralFilter(img_display, 9, blur_val * 10, 75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

        limit = contrast_clip / 10.0 if contrast_clip > 0 else 0.1
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        _, mask_dark = cv2.threshold(gray_enhanced, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        _, mask_gray_only = cv2.threshold(s_channel, max_sat, 255, cv2.THRESH_BINARY_INV)

        final_mask = cv2.bitwise_and(mask_dark, mask_gray_only)

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        display_mask = final_mask

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            if area > 80000: continue

            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 1 or y <= 1 or (x + w) >= w_img - 1 or (y + h) >= h_img - 1: continue

            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 1: aspect_ratio = 1.0 / aspect_ratio
            if aspect_ratio < remove_roads: continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            if solidity < solidity_thresh: continue

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(output, (x, y - 25), (x + 40, y), (0, 0, 0), -1)
            cv2.putText(output, str(count + 1), (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            count += 1

        cv2.rectangle(output, (0, 0), (220, 50), (0, 0, 0), -1)
        cv2.putText(output, f"Found: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow(window_name, output)

    # Виправляємо проблему "сірого вікна"
    mask_color = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Logic View', mask_color)


def start_processing(mode):
    global img_display, mode_current, window_name, trackbars_ready

    trackbars_ready = False
    mode_current = mode
    filename = FILE_LANDSAT if mode == 'low' else FILE_BING

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    original = cv2.imread(filename)
    if original is None: return

    h, w = original.shape[:2]

    if mode == 'low':
        scale = 10
        img_display = cv2.resize(original, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        window_name = "Landsat (Zoomed)"
    else:
        target_width = WINDOW_WIDTH
        scale = target_width / w if w > target_width else 1
        img_display = cv2.resize(original, (int(w * scale), int(h * scale)))
        window_name = "Bing Final Fix"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    cv2.namedWindow('Logic View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Logic View', 400, 300)

    if mode == 'low':
        cv2.createTrackbar('White Thresh', window_name, 190, 255, on_trackbar)
        cv2.createTrackbar('Black Thresh', window_name, 60, 255, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 2000, 10000, on_trackbar)
    else:
        cv2.createTrackbar('Texture Smooth', window_name, 12, 20, on_trackbar)
        cv2.createTrackbar('Contrast Boost', window_name, 2, 100, on_trackbar)
        cv2.createTrackbar('Dark Thresh', window_name, 182, 255, on_trackbar)
        cv2.createTrackbar('Max Saturation', window_name, 40, 255, on_trackbar)
        cv2.createTrackbar('Solidity %', window_name, 0, 100, on_trackbar)
        cv2.createTrackbar('Road Filter', window_name, 1, 100, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 404, 10000, on_trackbar)

    trackbars_ready = True
    on_trackbar(0)

    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("1 - Landsat\n2 - Bing\nChoice: ")
    if choice == '1':
        start_processing('low')
    else:
        start_processing('bing')