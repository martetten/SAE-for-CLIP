# test_hooks.py
# Цель: верификация корректности извлечения активаций через хуки
# Запуск: python experiments/test_hooks.py

import torch
import torch.nn as nn
from PIL import Image
import open_clip
import os

def hook_fn(name):
    """Универсальный хук, работающий с разными формами выхода"""
    def hook(module, input, output):
        # Обработка разных типов выхода
        tensor = output[0] if isinstance(output, tuple) else output

        if tensor.dim() == 3:  # [batch, seq_len, dim]
            cls_token = tensor[:, 0, :]
            print(f"[{name:20s}] Форма: {tensor.shape} → CLS: {cls_token.shape}")
        elif tensor.dim() == 2:  # [batch, dim] — уже агрегирован
            print(f"[{name:20s}] Форма: {tensor.shape} (агрегирован)")
        else:
            print(f"[{name:20s}] Неожиданная форма: {tensor.shape}")

        return output
    return hook

def create_test_image(size=(224, 224)):
    """Создаёт простое тестовое изображение (не требует интернета)"""
    img = Image.new('RGB', size, color=(73, 109, 137))
    # Добавим простую фигуру для разнообразия
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 50, 150, 150), fill=(255, 200, 0))
    return img

def main():
    # Загрузка модели CLIP
    print("Загрузка модели CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()
    model.cuda()
    print("✓ Модель загружена\n")

    # Создание тестового изображения (без интернета!)
    print("Создание тестового изображения...")
    img = create_test_image()
    img_tensor = preprocess(img).unsqueeze(0).cuda()
    print(f"✓ Изображение создано, форма тензора: {img_tensor.shape}\n")

    # Регистрация хуков только на блоках трансформера
    hooks = []
    print("Регистрация хуков на слоях трансформера...")
    for name, module in model.visual.named_modules():
        if 'resblocks' in name and isinstance(module, nn.Module):
            hook_handle = module.register_forward_hook(hook_fn(name))
            hooks.append(hook_handle)
    print(f"✓ Зарегистрировано хуков: {len(hooks)}\n")

    # Прямой проход
    print("Запуск прямого прохода...")
    with torch.no_grad():
        features = model.encode_image(img_tensor)
    print(f"\n✓ Финальный эмбеддинг: {features.shape}")

    # Удаление хуков
    for hook in hooks:
        hook.remove()
    print("\n✓ Все хуки удалены")

if __name__ == "__main__":
    main()