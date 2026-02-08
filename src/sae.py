"""
Архитектура разреженного автоэнкодера (SAE)
Реализация на основе методологии из репозитория saprmarks/dictionary_learning
"""

import torch
import torch.nn as nn
import math


class SparseAutoencoder(nn.Module):
    """
    Разреженный автоэнкодер с линейным энкодером/декодером и активацией Top-K.

    Архитектура:
        x (d_model) -> encoder (d_model → d_dict) -> relu -> topk -> decoder (d_dict → d_model) -> x_hat

    Где:
        d_model = 768 (размерность активаций CLIP)
        d_dict = d_model * expansion_factor (размер словаря)
    """

    def __init__(self, d_model=768, expansion_factor=64, k=500, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_model * expansion_factor
        self.k = k
        self.device = device if torch.cuda.is_available() else "cpu"

        # Энкодер БЕЗ встроенного биаса (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ)
        self.encoder = nn.Linear(d_model, self.d_dict, bias=False)  # bias=False

        # ЕДИНСТВЕННЫЙ источник смещения — отдельный параметр
        self.encoder_bias = nn.Parameter(torch.zeros(self.d_dict))

        # Декодер без биаса
        self.decoder = nn.Linear(self.d_dict, d_model, bias=False)
        self._init_decoder()

        self.to(self.device)

    def _init_decoder(self):
        """Инициализация декодера как ортонормированной матрицы"""
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            norm = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norm + 1e-8)

    def encode(self, x):
        """
        Энкодинг с ОДНИМ биасом до активации

        Args:
            x: [batch, d_model]

        Returns:
            latents: [batch, d_dict] — разреженные активации
        """
        # Линейное преобразование БЕЗ биаса
        pre_activations = self.encoder(x)  # [batch, d_dict]

        # Применение ЕДИНСТВЕННОГО биаса ДО нелинейности
        pre_activations = pre_activations + self.encoder_bias  # [batch, d_dict]

        # ReLU для неотрицательных активаций
        post_relu = torch.relu(pre_activations)

        # Top-K разреженность
        if self.k < self.d_dict:
            topk = torch.topk(post_relu, k=self.k, dim=-1)
            mask = torch.zeros_like(post_relu)
            mask.scatter_(-1, topk.indices, 1.0)
            latents = post_relu * mask
        else:
            latents = post_relu

        return latents

    def decode(self, latents):
        """Декодинг без биаса"""
        return self.decoder(latents)

    def forward(self, x):
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return x_hat, latents

    def normalize_decoder(self):
        """Нормализация столбцов декодера"""
        with torch.no_grad():
            norm = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norm + 1e-8)


# Тестовая функция
if __name__ == "__main__":
    print("Тестирование SparseAutoencoder...")
    sae = SparseAutoencoder(d_model=768, expansion_factor=64, k=500)

    batch_size = 8
    test_activations = torch.randn(batch_size, 768).to(sae.device)

    x_hat, latents = sae(test_activations)

    print(f" Вход: {test_activations.shape}")
    print(f" Реконструкция: {x_hat.shape}")
    print(f" Латенты: {latents.shape}")
    print(f" Размер словаря: {sae.d_dict}")

    l0 = (latents.abs() > 1e-6).float().sum(dim=-1).mean().item()
    print(f" Среднее L0 (активных фичей): {l0:.2f} (ожидалось ~500)")

    decoder_norm = sae.decoder.weight.norm(dim=0).mean().item()
    print(f" Средняя норма столбцов декодера: {decoder_norm:.4f} (ожидалось ~1.0)")

    # Проверка отсутствия двойного биаса
    assert sae.encoder.bias is None, "линейный слой содержит встроенный биас!"

    print("\nSparseAutoencoder готов к использованию!")