import cv2
import numpy as np
import os

# ==========================================
# ФАЙЛИ
# ==========================================
FILE_LANDSAT = "low.jpg"
FILE_BING = "bing.jpg"
# ==========================================

img_display = None
mode_current = None
window_name = None
trackbars_ready = False


def _safe_get(name: str, default: int) -> int:
    try:
        return cv2.getTrackbarPos(name, window_name)
    except cv2.error:
        return default


def on_trackbar(val):
    global img_display, mode_current, window_name, trackbars_ready
    if (not trackbars_ready) or (img_display is None):
        return

    output = img_display.copy()
    h_img, w_img = img_display.shape[:2]

    # ==========================================================
    # РЕЖИМ 1: LANDSAT (БЕЗ ЗМІН)
    # ==========================================================
    if mode_current == "low":
        thresh_white = _safe_get("White Thresh", 190)
        thresh_black = _safe_get("Black Thresh", 60)
        min_area = _safe_get("Min Area (Size)", 2000)

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

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5:
                    continue

                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 10)
                cx, cy = x + w // 2, y + h // 2
                cv2.putText(output, str(count + 1), (cx - 20, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                count += 1

        cv2.rectangle(output, (0, 0), (700, 150), (0, 0, 0), -1)
        cv2.putText(output, "AUTO-FILTER Active", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(output, f"Found: {count}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 8)

        cv2.imshow(window_name, output)
        cv2.imshow("Mask View", final_mask)
        return

    # ==========================================================
    # РЕЖИМ 2: BING (знаходить і "рожеві" будівлі, + фільтр зелені)
    # ==========================================================
    if mode_current == "bing":
        blur_smooth = _safe_get("Smooth Texture", 10)
        thresh_white = _safe_get("White Roofs", 200)
        thresh_gray = _safe_get("Gray Roofs", 110)
        min_area = _safe_get("Min Area", 1000)
        squareness = _safe_get("Remove Roads", 22) / 100.0

        # 0 = OFF, 1 = тільки навколо центральної будівлі, 2 = по всьому кадру
        green_filter_mode = _safe_get("Green Filter", 2)
        green_buffer = _safe_get("Green Buffer", 35)

        if blur_smooth < 1:
            blur_smooth = 1

        # 1) Згладжування + CLAHE
        smoothed = cv2.bilateralFilter(img_display, 9, blur_smooth * 10, 75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2) Маска "фону" за яскравістю (як було)
        _, mask_white = cv2.threshold(enhanced, thresh_white, 255, cv2.THRESH_BINARY)
        _, mask_dark = cv2.threshold(enhanced, thresh_gray, 255, cv2.THRESH_BINARY_INV)  # темні області
        combined_mask = cv2.bitwise_or(mask_white, mask_dark)

        # 3) Чистка фону
        k5 = np.ones((5, 5), np.uint8)
        combined_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, k5, iterations=1)
        combined_clean = cv2.morphologyEx(combined_clean, cv2.MORPH_CLOSE, k5, iterations=2)

        # 4) Інверсія (об'єкти = "дірки" у фоні)
        final_mask = cv2.bitwise_not(combined_clean)

        # 5) Маска зелені (зроблено трохи жорсткіше, щоб дахи не "зеленіли")
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 60, 40])
        upper_green = np.array([90, 255, 255])
        mask_grass = cv2.inRange(hsv, lower_green, upper_green)

        gk = np.ones((7, 7), np.uint8)
        mask_grass = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, gk, iterations=1)
        mask_grass = cv2.dilate(mask_grass, gk, iterations=1)

        # 6) Вирізаємо зелень ПІСЛЯ інверсії (щоб будівлі не ставали фоном)
        if green_filter_mode == 2:
            final_mask[mask_grass > 0] = 0

        elif green_filter_mode == 1:
            # знайти центральну будівлю і вирізати зелень тільки навколо неї
            contours0, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cx0, cy0 = w_img // 2, h_img // 2

            best_cnt = None
            best_d = 1e18

            for cnt in contours0:
                area = cv2.contourArea(cnt)
                if area < max(300, min_area):
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cxc = int(M["m10"] / M["m00"])
                cyc = int(M["m01"] / M["m00"])

                d = (cxc - cx0) ** 2 + (cyc - cy0) ** 2
                if d < best_d:
                    best_d = d
                    best_cnt = cnt

            if best_cnt is not None and green_buffer > 0:
                central_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.drawContours(central_mask, [best_cnt], -1, 255, thickness=-1)

                k = max(3, int(green_buffer))
                if k % 2 == 0:
                    k += 1
                bk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                buffer_mask = cv2.dilate(central_mask, bk, iterations=1)

                green_near = cv2.bitwise_and(mask_grass, buffer_mask)
                final_mask[green_near > 0] = 0

        # 7) КЛЮЧОВА ЗМІНА: повертаємо будівлі з "рожевих" квадратів
        # Вони інколи потрапляють у фон через пороги. Додаємо "дахові" кандидати окремо.
        roofs = cv2.bitwise_or(mask_white, mask_dark)
        roofs = cv2.bitwise_and(roofs, roofs, mask=cv2.bitwise_not(mask_grass))

        k_close = np.ones((5, 5), np.uint8)
        k_open = np.ones((3, 3), np.uint8)
        roofs = cv2.morphologyEx(roofs, cv2.MORPH_CLOSE, k_close, iterations=2)
        roofs = cv2.morphologyEx(roofs, cv2.MORPH_OPEN, k_open, iterations=1)

        final_mask = cv2.bitwise_or(final_mask, roofs)

        # 8) Контури + фільтр "квадратність"
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5:
                continue

            aspect_ratio = float(w) / h if h > 0 else 0.0
            if aspect_ratio > 1:
                aspect_ratio = 1.0 / aspect_ratio
            if aspect_ratio < squareness:
                continue

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, str(count + 1), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            count += 1

        cv2.rectangle(output, (0, 0), (260, 60), (0, 0, 0), -1)
        cv2.putText(output, "BING Mode", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Found: {count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(window_name, output)
        cv2.imshow("Mask View", final_mask)
        return


def start_processing(mode):
    global img_display, mode_current, window_name, trackbars_ready

    trackbars_ready = False
    mode_current = mode
    filename = FILE_LANDSAT if mode == "low" else FILE_BING

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    original = cv2.imread(filename)
    if original is None:
        print("Error: failed to read image.")
        return

    h, w = original.shape[:2]

    if mode == "low":
        scale = 10
        img_display = cv2.resize(original, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        window_name = "Landsat Auto-Clean"
    else:
        target_width = 800
        scale = target_width / w if w > target_width else 1
        img_display = cv2.resize(original, (int(w * scale), int(h * scale)))
        window_name = "Bing High-Res"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    cv2.namedWindow("Mask View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask View", 420, 320)

    if mode == "low":
        cv2.createTrackbar("White Thresh", window_name, 190, 255, on_trackbar)
        cv2.createTrackbar("Black Thresh", window_name, 60, 255, on_trackbar)
        cv2.createTrackbar("Min Area (Size)", window_name, 2000, 10000, on_trackbar)
    else:
        cv2.createTrackbar("Smooth Texture", window_name, 10, 20, on_trackbar)
        cv2.createTrackbar("White Roofs", window_name, 200, 255, on_trackbar)
        cv2.createTrackbar("Gray Roofs", window_name, 110, 255, on_trackbar)
        cv2.createTrackbar("Remove Roads", window_name, 22, 100, on_trackbar)
        cv2.createTrackbar("Min Area", window_name, 1000, 10000, on_trackbar)

        # Фільтр зелені
        cv2.createTrackbar("Green Filter", window_name, 2, 2, on_trackbar)  # 0..2
        cv2.createTrackbar("Green Buffer", window_name, 35, 250, on_trackbar)

    trackbars_ready = True
    on_trackbar(0)

    print("Press 'q' to exit.")
    while True:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("1 - Landsat (Auto)\n2 - Bing (Green + Roof fix)\nChoice: ")
    if choice == "1":
        start_processing("low")
    else:
        start_processing("bing")
