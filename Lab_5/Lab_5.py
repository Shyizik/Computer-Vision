import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import List, Optional, Dict

# Константи для налаштувань (легше змінювати в одному місці)
IMG_SIZE = (128, 128)
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200
WEIGHT_TEXTURE = 3.0
WEIGHT_WATER_INDEX = 5.0  # Збільшив вагу, як ми обговорювали раніше
WEIGHT_BLUE_CHANNEL = 5.0

# Мапінг категорій
CATEGORY_MAP = {
    'River': 'Water', 'SeaLake': 'Water',
    'Forest': 'Forest', 'Pasture': 'Vegetation',
    'Residential': 'Urban', 'Industrial': 'Industrial',
    'Highway': 'Roads',
    'AnnualCrop': 'Agriculture', 'PermanentCrop': 'Agriculture',
    'HerbaceousVegetation': 'Vegetation',
    'Forest_1': 'Forest', 'SeaLake_1': 'Water'
}


class GeoClusterer:
    """
    Клас для кластеризації супутникових знімків.
    Використовує комбінацію кольорових та текстурних ознак.
    """

    def __init__(self, data_folder: str, n_clusters: int = 5):
        self.data_folder = data_folder
        self.n_clusters = n_clusters
        self.image_paths: List[str] = []
        self._features: Optional[np.ndarray] = None
        self._scaler = StandardScaler()

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """Приватний метод для екстракції ознак з одного зображення."""

        # Ресайз для прискорення обробки
        img_resized = cv2.resize(img, IMG_SIZE)

        # 1. Гістограма HSV (Color Profile)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        # Hue: 12 бінів, Saturation: 4 біни.
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_features = hist.flatten()

        # 2. Середні значення каналів (RGB)
        # OpenCV завантажує в BGR
        mean_b = np.mean(img_resized[:, :, 0]) / 255.0
        mean_g = np.mean(img_resized[:, :, 1]) / 255.0
        # mean_r = np.mean(img_resized[:, :, 2]) / 255.0

        # 3. Текстура (Canny Edge Detection)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
        # Нормалізація щільності країв (0..1)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])

        # 4. Специфічні індекси (Domain Knowledge)
        water_index = (mean_b - mean_g)

        # Формування вектора ознак із застосуванням ваг
        features = np.concatenate([
            hist_features,
            [edge_density * WEIGHT_TEXTURE],
            [water_index * WEIGHT_WATER_INDEX],
            [mean_b * WEIGHT_BLUE_CHANNEL]
        ])

        return features

    def load_data(self) -> None:
        """Завантажує зображення та формує матрицю ознак."""
        # Підтримувані формати
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}

        # Пошук файлів (рекурсивно або ні - залежить від задачі, тут лінійно)
        all_files = glob.glob(os.path.join(self.data_folder, '*'))
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]

        print(f"[INFO] Знайдено {len(image_files)} зображень в '{self.data_folder}'.")

        features_list = []
        valid_paths = []

        for file_path in image_files:
            # imread не підтримує кирилицю в шляхах на Windows, тому краще так:
            # stream = open(file_path, "rb")
            # bytes = bytearray(stream.read())
            # numpyarray = np.asarray(bytes, dtype=np.uint8)
            # img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

            # Для спрощення залишимо imread, але з перевіркою
            img = cv2.imread(file_path)

            if img is None:
                print(f"[WARNING] Не вдалося прочитати файл: {file_path}")
                continue

            try:
                feat = self._extract_features(img)
                features_list.append(feat)
                valid_paths.append(file_path)
            except Exception as e:
                print(f"[ERROR] Помилка обробки {file_path}: {e}")

        if features_list:
            self._features = np.array(features_list)
            self.image_paths = valid_paths
            print(f"[INFO] Успішно оброблено {len(self.image_paths)} зображень.")
        else:
            print("[ERROR] Не сформовано жодного вектора ознак.")

    def run_clustering(self) -> Optional[np.ndarray]:
        """Запускає K-Means."""
        if self._features is None or len(self._features) < self.n_clusters:
            print(
                f"[ERROR] Недостатньо даних ({len(self.image_paths) if self.image_paths else 0}) для {self.n_clusters} кластерів.")
            return None

        # Масштабування - критично для K-Means
        scaled_features = self._scaler.fit_transform(self._features)

        # n_init='auto' або явне число (10-20) для стабільності
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(scaled_features)

        return labels

    def _get_smart_title(self, paths: List[str]) -> str:
        """Визначає домінуючу категорію в кластері."""
        votes = []
        for path in paths:
            filename = os.path.basename(path)
            # Шукаємо входження ключа в назву файлу
            found_category = "Unknown"
            for key, val in CATEGORY_MAP.items():
                if key in filename:
                    found_category = val
                    break
            votes.append(found_category)

        if not votes:
            return "Empty Cluster"

        # Повертає найчастіший елемент
        return Counter(votes).most_common(1)[0][0]

    def visualize_results(self, labels: np.ndarray) -> None:
        """Відображає результати."""
        if labels is None:
            return

        # Групування індексів
        clusters: Dict[int, List[str]] = {i: [] for i in range(self.n_clusters)}
        for path, label in zip(self.image_paths, labels):
            clusters[label].append(path)

        print(f"\n[RESULT] Результати кластеризації (K={self.n_clusters}):")

        for cluster_id, paths in clusters.items():
            if not paths:
                continue

            smart_title = self._get_smart_title(paths)
            count = len(paths)
            print(f" -> Кластер {cluster_id}: {smart_title} ({count} об'єктів)")

            # Обмеження кількості фото для показу
            n_show = min(count, 5)

            # Створення фігури
            fig, axes = plt.subplots(1, n_show, figsize=(15, 4))
            fig.canvas.manager.set_window_title(f"Cluster {cluster_id}")
            fig.suptitle(f"Cluster {cluster_id}: {smart_title}", fontsize=14, fontweight='bold')

            # Якщо лише 1 фото, axes не є списком
            if n_show == 1:
                axes = [axes]

            for i in range(n_show):
                # Конвертація кольору для matplotlib (BGR -> RGB)
                img = cv2.imread(paths[i])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                    axes[i].set_title(os.path.basename(paths[i]), fontsize=8)
                axes[i].axis('off')

            plt.show()
            # plt.close(fig) # Можна розкоментувати, якщо не потрібен блокуючий режим


if __name__ == "__main__":
    # Налаштування
    TARGET_CLUSTERS = 5
    # Краще використовувати абсолютний шлях або підпапку, щоб не сканувати код
    # Створи папку 'dataset_geo' і поклади фото туди
    DATA_DIR = "dataset_geo" if os.path.exists("dataset_geo") else "."

    geo_clusterer = GeoClusterer(data_folder=DATA_DIR, n_clusters=TARGET_CLUSTERS)

    try:
        geo_clusterer.load_data()

        # Перевірка чи є дані перед кластеризацією
        if geo_clusterer.image_paths:
            labels = geo_clusterer.run_clustering()
            geo_clusterer.visualize_results(labels)
        else:
            print("[Info] Додайте зображення у папку для початку роботи.")

    except KeyboardInterrupt:
        print("\n[Info] Роботу перервано користувачем.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Непередбачена помилка: {e}")