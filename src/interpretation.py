"""
Автоинтерпретация латентов через анализ активаций и генерацию коллажей
"""

import torch
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
from src.clip_utils import ClipActivationExtractor
from src.sae import SparseAutoencoder


class FeatureInterpreter:
    """
    Класс для извлечения топ-изображений по активации фичей и генерации коллажей
    """

    def __init__(self, sae_model, clip_extractor, device="cuda"):
        self.sae = sae_model
        self.clip_extractor = clip_extractor
        self.device = device
        self.sae.eval()

    def find_top_images(self, image_paths, feature_idx, top_k=12, batch_size=32):
        """
        Находит изображения с максимальной активацией для заданной фичи

        Args:
            image_paths: список путей к изображениям
            feature_idx: индекс фичи в словаре (0..49151)
            top_k: количество топ-изображений
            batch_size: размер батча для обработки

        Returns:
            top_paths: пути к топ-изображениям
            top_activations: значения активаций
        """
        all_activations = []
        all_paths = []

        # Обработка батчами
        for i in tqdm(range(0, len(image_paths), batch_size),
                    desc=f"  Фича {feature_idx:04d}", leave=False):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            # Загрузка и препроцессинг изображений
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.clip_extractor.preprocess_val(img).unsqueeze(0)
                    batch_images.append(tensor)
                except Exception as e:
                    continue

            if not batch_images:
                continue

            # Объединение в батч
            batch_tensor = torch.cat(batch_images, dim=0).to(self.device)

            # Извлечение активаций CLIP
            with torch.no_grad():
                clip_activations = self.clip_extractor.extract_activations(batch_tensor)
                # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: используем ТОЛЬКО энкодер для получения латентов
                latents = self.sae.encode(clip_activations)  # [batch, d_dict]

            # Сохранение активаций для целевой фичи
            feature_activations = latents[:, feature_idx].cpu().numpy()
            all_activations.append(feature_activations)
            all_paths.extend(batch_paths[:len(feature_activations)])

        # Объединение всех активаций
        all_activations = np.concatenate(all_activations)

        # Сортировка по убыванию активации
        sorted_indices = np.argsort(all_activations)[::-1]
        top_indices = sorted_indices[:top_k]

        return [all_paths[i] for i in top_indices], all_activations[top_indices]

    def create_collage(self, image_paths, output_path, grid_size=(4, 3), size=(224, 224)):
        """
        Создаёт коллаж 4x3 из изображений

        Args:
            image_paths: список путей к изображениям (12 штук)
            output_path: путь для сохранения коллажа
            grid_size: размер сетки (строки, столбцы)
            size: размер каждого изображения в коллаже
        """
        rows, cols = grid_size
        total_images = rows * cols

        # Создание холста
        collage_width = cols * size[0]
        collage_height = rows * size[1]
        collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

        # Добавление изображений в сетку
        for idx, img_path in enumerate(image_paths[:total_images]):
            try:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

                row = idx // cols
                col = idx % cols
                x = col * size[0]
                y = row * size[1]

                collage.paste(img, (x, y))
            except Exception as e:
                print(f"    Ошибка при обработке {img_path}: {e}")
                # Заполняем место серым квадратом при ошибке
                placeholder = Image.new('RGB', size, (200, 200, 200))
                row = idx // cols
                col = idx % cols
                x = col * size[0]
                y = row * size[1]
                collage.paste(placeholder, (x, y))

        # Сохранение коллажа
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        collage.save(output_path, quality=95)
        return output_path