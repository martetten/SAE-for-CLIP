"""
Работа с CLIP: загрузка модели, хуки для извлечения активаций со слоя 11 (последний слой трансформера)
"""

import torch
import open_clip
from PIL import Image
import torchvision.transforms as T


class ClipActivationExtractor:
    """
    Извлекает активации CLS-токена с указанного слоя трансформера CLIP.
    Для ViT-B/32 слой 11 — последний слой до финальной проекции в 512D.
    """

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", layer_idx=11, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.layer_idx = layer_idx

        # Загрузка модели
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Хранилище для активаций
        self.activations = None

        # Установка хука
        self._setup_hook()

    def _setup_hook(self):
        """Устанавливает хук на указанный слой трансформера для извлечения активаций"""
        # Очистка старых хуков
        for block in self.model.visual.transformer.resblocks:
            block._forward_hooks.clear()

        def hook_fn(module, input, output):
            # Обработка выхода: может быть тензор или кортеж
            x = output[0] if isinstance(output, tuple) else output

            # Формат активаций в OpenCLIP: [batch, seq_len, dim] = [batch, 50, 768]
            if x.ndim == 3:
                if x.shape[1] == 50:  # Формат [batch, seq_len, dim]
                    cls_token = x[:, 0, :]  # Извлекаем CLS-токен (первый в последовательности)
                else:
                    raise ValueError(f"Неожиданная длина последовательности: {x.shape[1]} (ожидалось 50)")
            else:
                raise ValueError(f"Неожиданная размерность тензора: {x.shape}")

            self.activations = cls_token.detach()

        # Установка хука на целевой слой
        target_block = self.model.visual.transformer.resblocks[self.layer_idx]
        target_block.register_forward_hook(hook_fn)

    def preprocess_image(self, image_path):
        """Загрузка и препроцессинг изображения"""
        img = Image.open(image_path).convert("RGB")
        return self.preprocess_val(img).unsqueeze(0).to(self.device)

    def extract_activations(self, images):
        """
        Извлекает активации со слоя 11 для батча изображений

        Args:
            images: torch.Tensor [batch, 3, 224, 224] или путь к изображению (строка)

        Returns:
            torch.Tensor [batch, 768] — активации CLS-токена
        """
        if isinstance(images, str):
            # Одиночное изображение по пути
            images = self.preprocess_image(images)

        with torch.no_grad():
            # Сброс предыдущих активаций
            self.activations = None
            # Прямой проход через модель
            _ = self.model.encode_image(images)

        if self.activations is None:
            raise RuntimeError("Активации не были извлечены. Проверьте работу хука.")

        return self.activations  # [batch, 768]

    def get_d_model(self):
        """Возвращает размерность активаций (768 для ViT-B/32)"""
        return 768


# Тестовая функция для верификации
if __name__ == "__main__":
    print("Тестирование ClipActivationExtractor...")
    extractor = ClipActivationExtractor(layer_idx=11, device="cuda" if torch.cuda.is_available() else "cpu")

    # Создание тестового изображения
    test_img = torch.randn(1, 3, 224, 224).to(extractor.device)

    # Извлечение активаций
    activations = extractor.extract_activations(test_img)

    print(f"Размерность активаций: {activations.shape}, должно быть: [1, 768]")
    print(f"mean: {activations.mean().item():.4f}")
    print(f"std: {activations.std().item():.4f}")
    print("\nClipActivationExtractor готов к использованию!")