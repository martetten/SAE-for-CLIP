import torch
import os

def clean_checkpoint(input_path, output_path):
    """Создаёт чистый чекпоинт только с весами модели"""
    print(f"Загрузка чекпоинта: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Извлечение ТОЛЬКО весов модели
    if 'model_state_dict' in checkpoint:
        clean_state_dict = checkpoint['model_state_dict']
    else:
        # Если структура другая — попробуем найти веса напрямую
        clean_state_dict = {k: v for k, v in checkpoint.items()
                           if not k.startswith(('optimizer', 'scheduler', 'epoch', 'metrics'))}

    # Сохранение чистого чекпоинта
    torch.save({
        'model_state_dict': clean_state_dict,
        'sae_config': checkpoint.get('sae_config', {}),
        'training_config': checkpoint.get('training_config', {}),
        'evr': checkpoint.get('metrics', {}).get('explained_variance', None)
    }, output_path)

    original_size = os.path.getsize(input_path) / 1024 / 1024
    clean_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nОригинальный размер: {original_size:.2f} МБ")
    print(f"Чистый размер: {clean_size:.2f} МБ")
    print(f"Сжатие: {(1 - clean_size/original_size)*100:.1f}%")
    print(f"\nЧистый чекпоинт сохранён: {output_path}")

if __name__ == "__main__":
    clean_checkpoint(
        "checkpoints_v2/sae_best.pth",
        "checkpoints_v2/sae_best_clean.pth"
    )