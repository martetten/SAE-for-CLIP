import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
CLI-скрипт для оценки влияния SAE на zero-shot классификацию
Использует два датасета разной сложности: CIFAR-10 (низкая) и CIFAR-100 (средняя)
"""

import argparse
import torch
import open_clip
from src.clip_with_sae import load_clip_with_sae
from src.evaluation import evaluate_zeroshot

# Подавление предупреждений о сетевых ошибках (для работы с кэшированными датасетами)
import warnings
warnings.filterwarnings("ignore", message=".*getaddrinfo failed.*")
warnings.filterwarnings("ignore", message=".*Using the latest cached version.*")
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")


def main():
    parser = argparse.ArgumentParser(description="Оценка влияния SAE на zero-shot классификацию")
    parser.add_argument("--sae_checkpoint", type=str,
                       default="./checkpoints_v2/sae_best.pth",
                       help="Путь к чекпоинту SAE")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32",
                       help="Архитектура CLIP")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k",
                       help="Претрейн-чекпоинт CLIP")
    parser.add_argument("--layer_idx", type=int, default=11,
                       help="Слой для интеграции SAE")
    parser.add_argument("--output", type=str, default="./data/zero_shot_metrics.md",
                       help="Путь для сохранения таблицы результатов")
    parser.add_argument("--cifar10_samples", type=int, default=None,
                       help="Количество образцов CIFAR-10 (None = все 10000)")
    parser.add_argument("--cifar100_samples", type=int, default=None,
                       help="Количество образцов CIFAR-100 (None = все 10000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}\n")

    # Загрузка оригинального CLIP
    print("Загрузка оригинального CLIP...")
    clip_original, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_original = clip_original.to(device).eval()

    # Загрузка CLIP с SAE
    print(f"Загрузка CLIP с SAE из {args.sae_checkpoint}...")
    clip_with_sae = load_clip_with_sae(
        args.sae_checkpoint,
        clip_model=args.clip_model,
        pretrained=args.clip_pretrained,
        layer_idx=args.layer_idx,
        device=device
    )
    clip_with_sae.eval()

    # Оценка на CIFAR-10
    print("Оценка на CIFAR-10 (низкая сложность, 10 классов, 32×32)")

    cifar10_top1_orig, cifar10_top5_orig = evaluate_zeroshot(
        clip_original, "cifar10", split="test",
        batch_size=128, num_samples=args.cifar10_samples,
        device=device, use_sae=False
    )

    cifar10_top1_sae, cifar10_top5_sae = evaluate_zeroshot(
        clip_with_sae, "cifar10", split="test",
        batch_size=128, num_samples=args.cifar10_samples,
        device=device, use_sae=True
    )

    cifar10_degradation_abs = cifar10_top1_orig - cifar10_top1_sae
    cifar10_degradation_rel = 100.0 * cifar10_degradation_abs / cifar10_top1_orig if cifar10_top1_orig > 0 else 0.0

    print(f"\nРезультаты CIFAR-10:")
    print(f"  Оригинальный CLIP:    Top-1: {cifar10_top1_orig:.2f}% | Top-5: {cifar10_top5_orig:.2f}%")
    print(f"  CLIP + SAE:           Top-1: {cifar10_top1_sae:.2f}% | Top-5: {cifar10_top5_sae:.2f}%")
    print(f"  Деградация Top-1:     {cifar10_degradation_abs:.2f} п.п. ({cifar10_degradation_rel:.2f}%)")

    # Оценка на CIFAR-100
    print("Оценка на CIFAR-100 (средняя сложность, 100 классов, 32×32)")

    cifar100_top1_orig, cifar100_top5_orig = evaluate_zeroshot(
        clip_original, "cifar100", split="test",
        batch_size=128, num_samples=args.cifar100_samples,
        device=device, use_sae=False
    )

    cifar100_top1_sae, cifar100_top5_sae = evaluate_zeroshot(
        clip_with_sae, "cifar100", split="test",
        batch_size=128, num_samples=args.cifar100_samples,
        device=device, use_sae=True
    )

    cifar100_degradation_abs = cifar100_top1_orig - cifar100_top1_sae
    cifar100_degradation_rel = 100.0 * cifar100_degradation_abs / cifar100_top1_orig if cifar100_top1_orig > 0 else 0.0

    print(f"\nРезультаты CIFAR-100:")
    print(f"  Оригинальный CLIP:    Top-1: {cifar100_top1_orig:.2f}% | Top-5: {cifar100_top5_orig:.2f}%")
    print(f"  CLIP + SAE:           Top-1: {cifar100_top1_sae:.2f}% | Top-5: {cifar100_top5_sae:.2f}%")
    print(f"  Деградация Top-1:     {cifar100_degradation_abs:.2f} п.п. ({cifar100_degradation_rel:.2f}%)")

    # Генерация таблицы для отчёта
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    table = f"""## Zero-Shot Классификация: Влияние SAE

| Датасет | Сложность | Модель | Top-1 Accuracy | Top-5 Accuracy | Абс. деградация | Отн. деградация |
|---------|-----------|--------|----------------|----------------|-----------------|-----------------|
| **CIFAR-10** | Низкая (10 классов,<br>32×32) | Оригинальный CLIP | {cifar10_top1_orig:.2f}% | {cifar10_top5_orig:.2f}% | — | — |
| | | CLIP + SAE | {cifar10_top1_sae:.2f}% | {cifar10_top5_sae:.2f}% | **{cifar10_degradation_abs:.2f} п.п.** | **{cifar10_degradation_rel:.2f}%** |
| **CIFAR-100** | Средняя (100 классов,<br>32×32) | Оригинальный CLIP | {cifar100_top1_orig:.2f}% | {cifar100_top5_orig:.2f}% | — | — |
| | | CLIP + SAE | {cifar100_top1_sae:.2f}% | {cifar100_top5_sae:.2f}% | **{cifar100_degradation_abs:.2f} п.п.** | **{cifar100_degradation_rel:.2f}%** |

"""

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(table)

    print(f"\n Таблица результатов сохранена: {args.output}")


if __name__ == "__main__":
    main()