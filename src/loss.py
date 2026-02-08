# Файл: src/loss.py
"""
Функции потерь и метрик для обучения разреженного автоэнкодера
"""

import torch
import torch.nn.functional as F


def mse_loss(x_hat, x):
    """Среднеквадратичная ошибка реконструкции"""
    return F.mse_loss(x_hat, x)


def l1_loss(latents):
    """L1 регуляризация для разреженности (сумма абсолютных значений активаций)"""
    return latents.abs().sum(dim=-1).mean()


def l0_approx(latents, threshold=1e-3):
    """
    Приближение L0 нормы (количество ненулевых активаций)
    Используется для метрики, не для градиентов
    """
    return (latents.abs() > threshold).float().sum(dim=-1).mean()


def explained_variance(x, x_hat):
    """
    Объяснённая дисперсия (аналог R^2)
    1.0 = идеальная реконструкция, 0.0 = хуже чем среднее
    """
    # Дисперсия ошибки
    error_var = (x - x_hat).var(dim=0).mean()
    # Дисперсия оригинала
    original_var = x.var(dim=0).mean()

    if original_var == 0:
        return torch.tensor(0.0, device=x.device)

    return 1 - (error_var / (original_var + 1e-8))


def combined_loss(x_hat, x, latents, sparsity_coeff=0.1):
    """
    Комбинированный лосс для обучения SAE

    Args:
        x_hat: реконструированные активации [batch, d_model]
        x: оригинальные активации [batch, d_model]
        latents: латентные коды [batch, d_dict]
        sparsity_coeff: коэффициент разреженности (λ в формуле λ·L1)

    Returns:
        total_loss: MSE + λ·L1
        metrics: словарь с компонентами лосса и метриками
    """
    mse = mse_loss(x_hat, x)

    # ВАЖНО: при топ-К активации L1 НЕ применяется (конфликт целей!)
    # Коэффициент sparsity_coeff игнорируется — разреженность контролируется параметром k
    total = mse

    metrics = {
        "mse_loss": mse.item(),
        "l1_loss": 0.0,  # Не используется при топ-К
        "total_loss": total.item(),
        "l0_approx": l0_approx(latents).item(),
        "explained_variance": explained_variance(x, x_hat).item()
    }

    return total, metrics


# Тестовая функция для верификации
if __name__ == "__main__":
    print("Тестирование функций потерь...")

    # Тестовые тензоры
    batch_size, d_model, d_dict = 8, 768, 24576
    x = torch.randn(batch_size, d_model)
    x_hat = x + torch.randn_like(x) * 0.1  # Немного зашумлённая реконструкция
    latents = torch.relu(torch.randn(batch_size, d_dict))
    latents[:, 100:] = 0  # Искусственная разреженность

    # Вычисление лоссов
    mse = mse_loss(x_hat, x)
    l1 = l1_loss(latents)
    l0 = l0_approx(latents)
    ev = explained_variance(x, x_hat)
    total, metrics = combined_loss(x_hat, x, latents, sparsity_coeff=0.1)

    print(f"MSE Loss: {mse.item():.6f}")
    print(f"L1 Loss: {l1.item():.6f}")
    print(f"L0 Approx (активных фичей): {l0.item():.2f}")
    print(f"Explained Variance: {ev.item():.4f}")
    print(f"Total Loss: {total.item():.6f}")
    print("\nФункции потерь готовы к использованию!")