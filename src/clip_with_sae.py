# src/clip_with_sae.py
"""
Интеграция SAE в прямой проход CLIP для оценки влияния на классификацию
"""

import torch
import open_clip
from src.sae import SparseAutoencoder


class ClipWithSAE:
    """
    Обёртка над CLIP, которая пропускает активации слоя 11 через SAE
    перед продолжением вычислений
    """

    def __init__(self, clip_model, sae_model, layer_idx=11, device="cuda"):
        self.clip = clip_model
        self.sae = sae_model
        self.layer_idx = layer_idx
        self.device = device
        self.activations = None

        # Установка хука для модификации активаций
        self._setup_hook()

    def _setup_hook(self):
        """Устанавливает хук для замены активаций через SAE"""
        # Очистка старых хуков
        for block in self.clip.visual.transformer.resblocks:
            block._forward_hooks.clear()

        def hook_fn(module, input, output):
            # Обработка выхода
            x = output[0] if isinstance(output, tuple) else output

            # Извлечение последовательности токенов
            if x.ndim == 3 and x.shape[1] == 50:  # [batch, seq_len, dim]
                # Пропускаем ВСЕ токены через SAE (не только CLS)
                batch_size, seq_len, dim = x.shape
                x_flat = x.view(-1, dim)  # [batch*seq_len, dim]

                # Реконструкция через SAE
                with torch.no_grad():
                    x_recon, _ = self.sae(x_flat)  # [batch*seq_len, dim]

                # Восстанавливаем форму последовательности
                x_modified = x_recon.view(batch_size, seq_len, dim)

                return x_modified

            return output

        # Установка хука на целевой слой
        target_block = self.clip.visual.transformer.resblocks[self.layer_idx]
        target_block.register_forward_hook(hook_fn)

    def encode_image(self, images, normalize=True):
        """Кодирование изображений с интеграцией SAE"""
        features = self.clip.encode_image(images, normalize=False)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, texts, normalize=True):
        """Кодирование текста (без изменений)"""
        features = self.clip.encode_text(texts, normalize=False)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    # Методы для совместимости с PyTorch
    def eval(self):
        self.clip.eval()
        self.sae.eval()
        return self

    def train(self):
        self.clip.train()
        self.sae.train()
        return self

    def to(self, device):
        self.clip = self.clip.to(device)
        self.sae = self.sae.to(device)
        self.device = device
        return self


def load_clip_with_sae(sae_checkpoint_path, clip_model="ViT-B-32",
                      pretrained="laion2b_s34b_b79k", layer_idx=11, device="cuda"):
    """
    Фабричная функция для загрузки модифицированной модели CLIP

    Args:
        sae_checkpoint_path: путь к чекпоинту SAE
        clip_model: архитектура CLIP
        pretrained: претрейн-чекпоинт
        layer_idx: слой для интеграции SAE

    Returns:
        ClipWithSAE instance
    """
    # Загрузка оригинального CLIP
    clip, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    clip = clip.to(device).eval()

    # Загрузка SAE
    checkpoint = torch.load(sae_checkpoint_path, map_location=device, weights_only=False)

    # Определение параметров SAE из чекпоинта
    if 'sae_config' in checkpoint:
        sae_config = checkpoint['sae_config']
    elif 'config' in checkpoint and 'sae' in checkpoint['config']:
        sae_config = checkpoint['config']['sae']
    else:
        sae_config = {
            'd_model': 768,
            'expansion_factor': 64,
            'k': 500
        }

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

    # Создание обёртки
    model = ClipWithSAE(clip, sae, layer_idx=layer_idx, device=device)
    return model