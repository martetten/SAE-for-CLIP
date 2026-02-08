import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
CLI-скрипт для автоинтерпретации 300+ латентов через анализ активаций
Генерирует коллажи ТОЛЬКО для самых активных фичей
"""

import argparse
import torch
import csv
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from src.sae import SparseAutoencoder
from src.clip_utils import ClipActivationExtractor


def load_food101_sample(sample_size=5000, seed=42):
    """Загружает случайную подвыборку изображений food101 для анализа"""
    from datasets import load_dataset
    import random

    print("Загрузка подвыборки food101 для анализа...")
    dataset = load_dataset("food101", split="train", streaming=False)

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    temp_dir = "./data/food101_sample"
    os.makedirs(temp_dir, exist_ok=True)

    image_paths = []
    for idx in tqdm(indices, desc="Сохранение изображений", leave=False):
        img = dataset[idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        path = os.path.join(temp_dir, f"img_{idx:06d}.jpg")
        img.save(path, quality=95)
        image_paths.append(path)

    print(f"  Сохранено {len(image_paths)} изображений в {temp_dir}")
    return image_paths


def analyze_all_features(sae, clip_extractor, image_paths, batch_size=32, device="cuda"):
    """
    Анализирует ВСЕ фичи на подвыборке и возвращает топ-N самых активных

    Возвращает:
        top_feature_indices: список индексов топ-фичей
        feature_stats: словарь {feature_idx: {mean_activation, max_activation, top_images}}
    """
    print(f"\nАнализ всех {sae.d_dict} фичей на {len(image_paths)} изображениях...")

    # Накопление активаций для всех фичей
    all_activations = np.zeros((len(image_paths), sae.d_dict), dtype=np.float32)

    # Обработка батчами
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Извлечение активаций"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = clip_extractor.preprocess_val(img).unsqueeze(0)
                batch_images.append(tensor)
            except Exception as e:
                continue

        if not batch_images:
            continue

        batch_tensor = torch.cat(batch_images, dim=0).to(device)

        with torch.no_grad():
            clip_activations = clip_extractor.extract_activations(batch_tensor)
            latents = sae.encode(clip_activations)  # [batch, d_dict]
            all_activations[i:i+len(batch_images)] = latents.cpu().numpy()

    # Расчёт статистик для каждой фичи
    feature_means = all_activations.mean(axis=0)
    feature_maxs = all_activations.max(axis=0)

    # Сортировка по средней активации (убывание)
    sorted_indices = np.argsort(feature_means)[::-1]

    # Отбор топ-300 фичей
    top_indices = sorted_indices[:300]

    # Сбор топ-изображений для каждой фичи
    feature_stats = {}
    for feature_idx in tqdm(top_indices, desc="Поиск топ-изображений"):
        # Сортировка изображений по активации фичи
        image_indices = np.argsort(all_activations[:, feature_idx])[::-1][:12]
        top_paths = [image_paths[idx] for idx in image_indices]
        top_activations = all_activations[image_indices, feature_idx]

        feature_stats[int(feature_idx)] = {
            "mean_activation": float(feature_means[feature_idx]),
            "max_activation": float(feature_maxs[feature_idx]),
            "top_paths": top_paths,
            "top_activations": top_activations.tolist()
        }

    return top_indices.tolist(), feature_stats


def create_collage(image_paths, output_path, grid_size=(4, 3), size=(224, 224)):
    """Создаёт коллаж 4x3 из изображений"""
    rows, cols = grid_size
    collage_width = cols * size[0]
    collage_height = rows * size[1]
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    for idx, img_path in enumerate(image_paths[:rows*cols]):
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            row = idx // cols
            col = idx % cols
            x = col * size[0]
            y = row * size[1]
            collage.paste(img, (x, y))
        except Exception as e:
            placeholder = Image.new('RGB', size, (200, 200, 200))
            row = idx // cols
            col = idx % cols
            x = col * size[0]
            y = row * size[1]
            collage.paste(placeholder, (x, y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    collage.save(output_path, quality=95)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Автоинтерпретация латентов SAE")
    parser.add_argument("--sae_checkpoint", type=str,
                       default="./checkpoints_v2/sae_best.pth",
                       help="Путь к чекпоинту SAE")
    parser.add_argument("--top_k", type=int, default=12,
                       help="Количество топ-изображений на фичу")
    parser.add_argument("--sample_size", type=int, default=5000,
                       help="Размер подвыборки изображений для анализа")
    parser.add_argument("--output_csv", type=str, default="./data/interpretations.csv",
                       help="Путь для сохранения результатов")
    parser.add_argument("--collage_dir", type=str, default="./assets/report_collages",
                       help="Директория для сохранения коллажей")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}\n")

    # Загрузка подвыборки
    image_paths = load_food101_sample(sample_size=args.sample_size)

    # Загрузка моделей
    print("\nЗагрузка моделей...")
    clip_extractor = ClipActivationExtractor(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        layer_idx=11,
        device=device
    )

    checkpoint = torch.load(args.sae_checkpoint, map_location=device, weights_only=False)
    sae_config = checkpoint.get('sae_config', {})
    d_model = sae_config.get('d_model', 768)
    expansion_factor = sae_config.get('expansion_factor', 64)
    k = sae_config.get('k', 500)

    sae = SparseAutoencoder(
        d_model=d_model,
        expansion_factor=expansion_factor,
        k=k,
        device=device
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    # Анализ ВСЕХ фичей и отбор топ-300
    top_feature_indices, feature_stats = analyze_all_features(
        sae, clip_extractor, image_paths, batch_size=32, device=device
    )

    # Генерация коллажей и сохранение результатов
    print(f"\nГенерация коллажей для топ-300 фичей...")
    results = []

    for rank, feature_idx in enumerate(tqdm(top_feature_indices, desc="Генерация коллажей")):
        stats = feature_stats[feature_idx]

        # Генерация коллажа
        collage_path = os.path.join(args.collage_dir, f"feature_{feature_idx:05d}.jpg")
        create_collage(stats["top_paths"], collage_path, grid_size=(4, 3), size=(224, 224))

        # Сохранение результата
        results.append({
            "feature_rank": rank + 1,  # Ранг по активности (1 = самая активная)
            "feature_id": feature_idx,
            "collage_path": collage_path,
            "activation_mean": stats["mean_activation"],
            "activation_max": stats["max_activation"],
            "num_images": len(stats["top_paths"]),
            "interpretation": ""  # Заполнится вручную
        })

    # Сохранение результатов
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Результаты сохранены: {args.output_csv}")
    print(f" Коллажи сохранены в: {args.collage_dir}")
    print(f"\nСтатистика:")
    print(f"  Всего фичей в словаре: {sae.d_dict:,}")
    print(f"  Проанализировано изображений: {len(image_paths)}")
    print(f"  Отобрано топ-фичей: {len(results)}")
    print(f"  Средняя активация топ-фичи: {results[0]['activation_mean']:.3f}")

if __name__ == "__main__":
    main()