import cv2
import numpy as np
import os

# ==========================================
# FILE CONFIGURATION
# ==========================================
FILE_LANDSAT = 'low.jpg'
FILE_BING = 'bing.jpg'
# ==========================================

img_display = None
mode_current = None
window_name = None
trackbars_ready = False


def on_trackbar(val):
    """
    Main processing loop triggered by UI updates.
    """
    if not trackbars_ready or img_display is None: return

    output = img_display.copy()
    h_img, w_img = img_display.shape[:2]

    # =========================================================
    # MODE 1: LANDSAT (Dark Object Detection)
    # =========================================================
    if mode_current == 'low':
        # UI Parameters
        blur_val = cv2.getTrackbarPos('Blur', window_name)
        thresh_val = cv2.getTrackbarPos('Threshold', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)
        contrast_val = cv2.getTrackbarPos('Contrast', window_name)

        # Parameter validation
        if blur_val % 2 == 0: blur_val += 1
        if blur_val < 1: blur_val = 1

        # 1. Preprocessing (Grayscale + CLAHE + Median Blur)
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        clip_limit = contrast_val / 10.0 if contrast_val > 0 else 0.1
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced, blur_val)

        # 2. Binary Thresholding (Inverse)
        # Inverts mask: Dark objects become White (255), Light background becomes Black (0)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # 3. Morphological Operations (Noise Reduction)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Contour Detection
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Area Filtering
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)

                # Boundary Check (Ignore image borders)
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5: continue

                # Visualization
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, str(count + 1), (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                count += 1

        # Status Display
        cv2.rectangle(output, (0, 0), (350, 50), (0, 0, 0), -1)
        cv2.putText(output, f"Objects Detected: {count}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', binary)

    # ==========================================================
    # MODE 2: BING (Geometric Structure Detection)
    # ==========================================================
    elif mode_current == 'bing':
        # UI Parameters
        c_min = cv2.getTrackbarPos('Canny Min', window_name)
        c_max = cv2.getTrackbarPos('Canny Max', window_name)
        thick = cv2.getTrackbarPos('Thickness', window_name)
        min_area = cv2.getTrackbarPos('Min Area', window_name)

        if thick < 1: thick = 1

        # 1. Preprocessing
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. Edge Detection (Canny)
        edges = cv2.Canny(blurred, c_min, c_max)

        # 3. Morphology (Connecting gaps)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.dilate(edges, kernel, iterations=thick)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. Contour Detection
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Area Filtering
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 5 or y <= 5 or (x + w) >= w_img - 5 or (y + h) >= h_img - 5: continue

                # Polygon Approximation
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Visualization
                cv2.drawContours(output, [approx], 0, (0, 255, 0), 2)
                cv2.putText(output, str(count + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                count += 1

        # Status Display
        cv2.putText(output, f"Buildings Found: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, output)
        cv2.imshow('Mask View', closed)


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

    # Window Setup and Scaling
    h, w = original.shape[:2]
    if mode == 'low':
        scale = 10
        img_display = cv2.resize(original, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        window_name = "Landsat Analysis"
    else:
        target_width = 1000
        scale = target_width / w if w > target_width else 1
        img_display = cv2.resize(original, (int(w * scale), int(h * scale)))
        window_name = "Bing Analysis"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)
    cv2.namedWindow('Mask View', cv2.WINDOW_NORMAL)

    # Trackbar Initialization
    if mode == 'low':
        cv2.createTrackbar('Blur', window_name, 7, 30, on_trackbar)
        cv2.createTrackbar('Threshold', window_name, 80, 255, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 500, 5000, on_trackbar)
        cv2.createTrackbar('Contrast', window_name, 40, 100, on_trackbar)
    else:
        cv2.createTrackbar('Canny Min', window_name, 50, 255, on_trackbar)
        cv2.createTrackbar('Canny Max', window_name, 150, 255, on_trackbar)
        cv2.createTrackbar('Thickness', window_name, 2, 10, on_trackbar)
        cv2.createTrackbar('Min Area', window_name, 500, 5000, on_trackbar)

    trackbars_ready = True
    on_trackbar(0)

    print("System ready. Press 'q' to exit.")
    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("Select Mode (1-Landsat, 2-Bing): ")
    if choice == '1':
        start_processing('low')
    else:
        start_processing('bing')