import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import List, Optional, Dict

# Налаштування параметрів обробки
IMG_SIZE = (128, 128)
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200
WEIGHT_TEXTURE = 3.0
WEIGHT_WATER_INDEX = 5.0
WEIGHT_BLUE_CHANNEL = 5.0

# Словник для спрощення назв категорій
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
    """Клас для кластеризації зображень за кольором та текстурою."""

    def __init__(self, data_folder: str, n_clusters: int = 5):
        self.data_folder = data_folder
        self.n_clusters = n_clusters
        self.image_paths: List[str] = []
        self._features: Optional[np.ndarray] = None
        self._scaler = StandardScaler()

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """Вилучає вектор числових ознак з зображення."""

        # Зменшення розміру для швидкодії
        img_resized = cv2.resize(img, IMG_SIZE)

        # 1. Аналіз кольору (HSV гістограма)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_features = hist.flatten()

        # 2. Середні значення каналів (BGR)
        mean_b = np.mean(img_resized[:, :, 0]) / 255.0
        mean_g = np.mean(img_resized[:, :, 1]) / 255.0

        # 3. Аналіз текстури (детектор меж Canny)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])

        # 4. Специфічний індекс для виявлення води
        water_index = (mean_b - mean_g)

        # Збірка фінального вектора ознак із вагами
        features = np.concatenate([
            hist_features,
            [edge_density * WEIGHT_TEXTURE],
            [water_index * WEIGHT_WATER_INDEX],
            [mean_b * WEIGHT_BLUE_CHANNEL]
        ])

        return features

    def load_data(self) -> None:
        """Зчитує зображення з папки та готує дані."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}

        # Отримання списку файлів
        all_files = glob.glob(os.path.join(self.data_folder, '*'))
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]

        print(f"[INFO] Знайдено {len(image_files)} зображень.")

        features_list = []
        valid_paths = []

        for file_path in image_files:
            img = cv2.imread(file_path)

            if img is None:
                print(f"[WARNING] Помилка читання: {file_path}")
                continue

            try:
                # Обчислення ознак для кожного фото
                feat = self._extract_features(img)
                features_list.append(feat)
                valid_paths.append(file_path)
            except Exception as e:
                print(f"[ERROR] Збій обробки {file_path}: {e}")

        if features_list:
            self._features = np.array(features_list)
            self.image_paths = valid_paths
            print(f"[INFO] Оброблено {len(self.image_paths)} зображень.")
        else:
            print("[ERROR] Дані відсутні.")

    def run_clustering(self) -> Optional[np.ndarray]:
        """Виконує нормалізацію даних та алгоритм K-Means."""
        if self._features is None or len(self._features) < self.n_clusters:
            print(f"[ERROR] Недостатньо даних для кластеризації.")
            return None

        # Стандартизація даних (критично для коректної роботи дистанцій)
        scaled_features = self._scaler.fit_transform(self._features)

        # Кластеризація
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(scaled_features)

        return labels

    def _get_smart_title(self, paths: List[str]) -> str:
        """Визначає назву кластера за найчастішою категорією у назвах файлів."""
        votes = []
        for path in paths:
            filename = os.path.basename(path)
            found_category = "Unknown"
            # Пошук підстроки категорії у назві файлу
            for key, val in CATEGORY_MAP.items():
                if key in filename:
                    found_category = val
                    break
            votes.append(found_category)

        if not votes:
            return "Empty Cluster"

        return Counter(votes).most_common(1)[0][0]

    def visualize_results(self, labels: np.ndarray) -> None:
        """Візуалізує вибірку зображень для кожного кластера."""
        if labels is None:
            return

        # Групування шляхів за ID кластера
        clusters: Dict[int, List[str]] = {i: [] for i in range(self.n_clusters)}
        for path, label in zip(self.image_paths, labels):
            clusters[label].append(path)

        print(f"\n[RESULT] Результати (K={self.n_clusters}):")

        for cluster_id, paths in clusters.items():
            if not paths:
                continue

            smart_title = self._get_smart_title(paths)
            count = len(paths)
            print(f" -> Кластер {cluster_id}: {smart_title} ({count} фото)")

            # Відображення до 5 зображень з кластера
            n_show = min(count, 5)
            fig, axes = plt.subplots(1, n_show, figsize=(15, 4))
            fig.suptitle(f"Cluster {cluster_id}: {smart_title}", fontsize=14, fontweight='bold')

            if n_show == 1:
                axes = [axes]

            for i in range(n_show):
                img = cv2.imread(paths[i])
                if img is not None:
                    # Конвертація BGR -> RGB для коректного відображення в Matplotlib
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                    axes[i].set_title(os.path.basename(paths[i]), fontsize=8)
                axes[i].axis('off')

            plt.show()


if __name__ == "__main__":
    TARGET_CLUSTERS = 5
    # Використання поточної директорії або 'dataset_geo'
    DATA_DIR = "dataset_geo" if os.path.exists("dataset_geo") else "."

    geo_clusterer = GeoClusterer(data_folder=DATA_DIR, n_clusters=TARGET_CLUSTERS)

    try:
        geo_clusterer.load_data()

        if geo_clusterer.image_paths:
            labels = geo_clusterer.run_clustering()
            geo_clusterer.visualize_results(labels)
        else:
            print("[Info] Папка порожня. Додайте зображення.")

    except KeyboardInterrupt:
        print("\n[Info] Скасовано користувачем.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Помилка: {e}")