import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
CLI точка входа для обучения SAE на активациях CLIP
Поддерживает загрузку конфигурации из двух YAML файлов:
  - sae_config.yaml: гиперпараметры модели
  - training_config.yaml: параметры обучения
"""

import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from src.clip_utils import ClipActivationExtractor
from src.sae import SparseAutoencoder
from src.training_loop import train_sae


def collate_images(batch):
    """Именованная функция коллатора (безопасна для пиклинга в Windows)"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    return {"image": pixel_values}


def load_yaml(path):
    """Загрузка YAML конфигурации"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_transforms():
    """Трансформы, совместимые с OpenCLIP"""
    return Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
    ])


def get_image_column(dataset):
    """Автоопределение колонки с изображением"""
    columns = dataset.column_names
    if 'image' in columns:
        return 'image'
    elif 'img' in columns:
        return 'img'
    else:
        raise ValueError(f"Не найдена колонка с изображением. Колонки: {columns}")


def main():
    parser = argparse.ArgumentParser(description="Обучение SAE для CLIP")
    parser.add_argument("--sae_config", type=str, default="configs/sae_config.yaml",
                       help="Путь к конфигурации SAE")
    parser.add_argument("--training_config", type=str, default="configs/training_config.yaml",
                       help="Путь к конфигурации обучения")
    args = parser.parse_args()

    # Загрузка конфигураций
    sae_cfg = load_yaml(args.sae_config)
    train_cfg = load_yaml(args.training_config)
    print(f"Загружены конфигурации:\n  SAE: {args.sae_config}\n  Обучение: {args.training_config}")

    # Устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    # Загрузка датасета
    print(f"\nЗагрузка {train_cfg['dataset']['name']} ({train_cfg['dataset']['split']})...")
    dataset = load_dataset(
        train_cfg['dataset']['name'],
        split=train_cfg['dataset']['split'],
        token=True  # использует токен из переменной окружения HF_TOKEN
    )

    # Определение колонки изображения
    image_column = get_image_column(dataset)
    print(f"Колонка изображения: '{image_column}'")

    # Применение трансформов
    transforms = get_transforms()

    def apply_transforms(example):
        try:
            img = example[image_column]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return {"pixel_values": transforms(img)}
        except Exception:
            return None

    dataset = dataset.map(
        apply_transforms,
        remove_columns=dataset.column_names,
        desc="Трансформация изображений"
    ).filter(lambda x: x is not None and x.get("pixel_values") is not None)

    dataset = dataset.with_format("torch", columns=["pixel_values"])
    print(f"Датасет готов: {len(dataset)} изображений")

    # Даталоадер
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['dataset']['batch_size'],
        shuffle=True,
        num_workers=train_cfg['dataset']['num_workers'],
        pin_memory=True if device == "cuda" else False,
        collate_fn=collate_images
    )

    # Инициализация моделей
    print(f"\nИнициализация CLIP {sae_cfg['clip']['model']} (слой {sae_cfg['clip']['layer']})...")
    extractor = ClipActivationExtractor(
        model_name=sae_cfg['clip']['model'],
        pretrained=sae_cfg['clip']['pretrained'],
        layer_idx=sae_cfg['clip']['layer'],
        device=device
    )

    print(f"Инициализация SAE (expansion_factor={sae_cfg['sae']['expansion_factor']}, k={sae_cfg['sae']['k']})...")
    sae = SparseAutoencoder(
        d_model=sae_cfg['sae']['d_model'],
        expansion_factor=sae_cfg['sae']['expansion_factor'],
        k=sae_cfg['sae']['k'],
        device=device
    )

    # Обучение
    print("Обучение SAE")
    print(f"\nДатасет: {train_cfg['dataset']['name']} ({len(dataset)} изображений)")
    print(f"Батч: {train_cfg['dataset']['batch_size']}, Эпохи: {train_cfg['training']['num_epochs']}")
    print(f"Размер словаря: {sae_cfg['sae']['d_model'] * sae_cfg['sae']['expansion_factor']:,} фич")
    print(f"Целевая разреженность: топ-{sae_cfg['sae']['k']} фичей")

    trained_sae, log_path = train_sae(
        sae=sae,
        activation_extractor=extractor,
        dataloader=dataloader,
        num_epochs=train_cfg['training']['num_epochs'],
        lr=train_cfg['training']['lr'],
        sparsity_coeff=sae_cfg['sae']['sparsity_coeff'],
        log_dir=train_cfg['training']['log_dir'],
        device=device,
        checkpoint_dir=train_cfg['training']['checkpoint_dir'],
        save_every_epoch=train_cfg['training']['save_every_epoch']
    )

    # Сохранение финальной модели
    os.makedirs(train_cfg['training']['checkpoint_dir'], exist_ok=True)
    final_path = os.path.join(
        train_cfg['training']['checkpoint_dir'],
        f"sae_clip_vitb32_layer{sae_cfg['clip']['layer']}_ef{sae_cfg['sae']['expansion_factor']}.pth"
    )

    torch.save({
        'model_state_dict': trained_sae.state_dict(),
        'sae_config': sae_cfg,
        'training_config': train_cfg,
        'log_path': log_path,
        'metrics': {
            'dict_size': sae_cfg['sae']['d_model'] * sae_cfg['sae']['expansion_factor'],
            'target_l0': sae_cfg['sae']['k']
        }
    }, final_path)

    print(f"\nМодель сохранена в: {final_path}")
    print("\nДля анализа метрик выполните:")
    print(f"python scripts/analyze_training.py --logdir {log_path}")
    print("\nДля запуска TensorBoard:")
    print(f"tensorboard --logdir {train_cfg['training']['log_dir']}")


if __name__ == "__main__":
    main()