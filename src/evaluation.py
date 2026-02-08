"""
Оценка zero-shot классификации с/без SAE
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import open_clip


def get_clip_preprocess():
    """Возвращает препроцессинг, совместимый с OpenCLIP"""
    return Compose([
        Resize(224, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
    ])


def get_cifar10_prompts():
    """Стандартные промпты для CIFAR-10 из статьи CLIP"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    return [f"a photo of a {c}" for c in classes], classes


def get_cifar100_prompts():
    """Стандартные промпты для CIFAR-100 из статьи CLIP"""
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    classes_readable = [c.replace('_', ' ') for c in classes]
    return [f"a photo of a {c}" for c in classes_readable], classes_readable


def evaluate_zeroshot(model, dataset_name, split="test", batch_size=64,
                     num_samples=None, device="cuda", use_sae=False):
    """
    Оценка zero-shot классификации

    Args:
        model: ClipWithSAE или оригинальная модель CLIP
        dataset_name: "cifar10" или "cifar100"
        split: сплит датасета
        batch_size: размер батча
        num_samples: ограничение на количество образцов
        device: устройство
        use_sae: флаг для корректного вызова методов

    Returns:
        top1_acc: точность Top-1 (%)
        top5_acc: точность Top-5 (%)
    """
    # Загрузка датасета и определение структуры
    print(f"  Загрузка {dataset_name} ({split})...")
    if dataset_name == "cifar10":
        dataset = load_dataset(dataset_name, split=split)
        prompts, _ = get_cifar10_prompts()
        image_column = "img"
        label_column = "label"
    elif dataset_name == "cifar100":
        dataset = load_dataset(dataset_name, split=split)
        prompts, _ = get_cifar100_prompts()
        image_column = "img"
        label_column = "fine_label"  # ← В CIFAR-100 метка хранится в 'fine_label'
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    # Подготовка текстовых фичей
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text_tokens = tokenizer(prompts).to(device)

    if use_sae:
        text_features = model.encode_text(text_tokens)
    else:
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

    # Подготовка изображений с правильной обработкой меток
    transforms = get_clip_preprocess()

    # ВАЖНО: определяем функцию трансформации ВНУТРИ с доступом к label_column
    def apply_transforms(example):
        img = example[image_column]
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Используем ПРАВИЛЬНУЮ колонку метки в зависимости от датасета
        return {
            "pixel_values": transforms(img),
            "label": example[label_column]  # ← Теперь корректно работает для обоих датасетов
        }

    # Применяем трансформы
    dataset = dataset.map(
        apply_transforms,
        remove_columns=[col for col in dataset.column_names if col not in ["pixel_values", "label"]],
        desc="Трансформация изображений"
    )
    dataset = dataset.with_format("torch", columns=["pixel_values", "label"])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
        collate_fn=lambda batch: {
            "images": torch.stack([item["pixel_values"] for item in batch]),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long)
        }
    )

    # Оценка
    print(f"  Оценка на {len(dataset)} изображениях...")
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    {dataset_name}", leave=False):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            if use_sae:
                image_features = model.encode_image(images, normalize=True)
            else:
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T

            # Top-1
            preds = logits.argmax(dim=1)
            top1_correct += (preds == labels).sum().item()

            # Top-5
            top5_preds = logits.topk(5, dim=1).indices
            top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    return top1_acc, top5_acc