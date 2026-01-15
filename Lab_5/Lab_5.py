import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import glob
from collections import Counter

# Словник для семантичного маппингу назв файлів у категорії
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
    Клас для автоматичної кластеризації супутникових знімків
    на основі комп'ютерного зору (Feature Extraction + K-Means).
    """

    def __init__(self, data_folder, n_clusters=5):
        """
        Ініціалізація параметрів.
        :param data_folder: Шлях до директорії з даними.
        :param n_clusters: Цільова кількість кластерів.
        """
        self.data_folder = data_folder
        self.n_clusters = n_clusters
        self.image_paths = []
        self.features = []

    def extract_features(self, img):
        """
        Вилучення вектора ознак.
        Штраф за червоний канал (mean_r) прибрано.
        """
        # Стандартизація розміру
        img = cv2.resize(img, (128, 128))

        # 1. Гістограма кольорів у просторі HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_features = hist.flatten()

        # 2. Розрахунок середніх значень каналів (Blue/Green)
        mean_b = np.mean(img[:, :, 0]) / 255.0
        mean_g = np.mean(img[:, :, 1]) / 255.0
        # mean_r розраховувати не обов'язково, оскільки ми його виключили

        # 3. Аналіз текстури (Edge Detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1]) / 255.0

        # 4. Water Index (Синій проти Зеленого)
        water_index = (mean_b - mean_g) * 5.0

        # Формування фінального вектора
        features = np.concatenate([
            hist_features,  # Загальний колір
            [edge_density * 3.0],  # Текстура (міста)
            [water_index],  # Індекс води
            [mean_b * 5.0]  # Синій канал (для води)
        ])

        return features

    def load_data(self):
        """Зчитування зображень та формування dataset."""
        extensions = ['*.jpg', '*.jpeg', '*.png']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.data_folder, ext)))

        print(f"[INFO] Знайдено {len(files)} зображень. Початок обробки...")

        valid_features = []
        valid_paths = []

        for file in files:
            img = cv2.imread(file)
            if img is not None:
                feat = self.extract_features(img)
                valid_features.append(feat)
                valid_paths.append(file)

        self.features = np.array(valid_features)
        self.image_paths = valid_paths

    def run_clustering(self):
        """Виконання кластеризації методом K-Means."""
        if len(self.features) < self.n_clusters:
            print("[ERROR] Недостатньо даних для кластеризації.")
            return None

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(scaled_features)
        return labels

    def get_smart_title(self, paths):
        """Визначає семантичну назву кластера."""
        votes = []
        for path in paths:
            filename = os.path.basename(path)
            found = "Unknown"
            for key, val in CATEGORY_MAP.items():
                if key in filename:
                    found = val
                    break
            votes.append(found)

        if not votes: return "Unknown"
        most_common, _ = Counter(votes).most_common(1)[0]
        return most_common

    def visualize_results(self, labels):
        """Візуалізація результатів."""
        if labels is None: return

        clusters = {i: [] for i in range(self.n_clusters)}
        for path, label in zip(self.image_paths, labels):
            clusters[label].append(path)

        print(f"\n[RESULT] Результати кластеризації (K={self.n_clusters}):")

        for cluster_id, paths in clusters.items():
            if not paths: continue

            smart_title = self.get_smart_title(paths)
            count = len(paths)
            print(f" -> Кластер {cluster_id}: {smart_title} ({count} об'єктів)")

            n_show = min(count, 5)
            fig, axes = plt.subplots(1, n_show, figsize=(15, 5))

            full_title = f"Cluster {cluster_id}: {smart_title}"
            fig.canvas.manager.set_window_title(full_title)
            fig.suptitle(full_title, fontsize=14, fontweight='bold')

            if n_show == 1: axes = [axes]

            for i in range(n_show):
                img = cv2.imread(paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(os.path.basename(paths[i]), fontsize=9)

            plt.show()


if __name__ == "__main__":
    TARGET_CLUSTERS = 5

    try:
        clusterer = GeoClusterer(data_folder=".", n_clusters=TARGET_CLUSTERS)
        clusterer.load_data()

        if len(clusterer.image_paths) > 0:
            labels = clusterer.run_clustering()
            clusterer.visualize_results(labels)
        else:
            print("[WARNING] Зображення не знайдено.")
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")