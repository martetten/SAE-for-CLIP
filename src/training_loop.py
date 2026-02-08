"""
Тренировочный цикл с логгированием через TensorBoard
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from tqdm import tqdm

from src.loss import combined_loss

class SAEScheduler:
    """
    Scheduler для коэффициента разреженности (аналогично dictionary_learning)
    Начинает с низкого значения, постепенно увеличивает до целевого
    """

    def __init__(self, initial_value=1e-4, final_value=0.1, warmup_steps=1000):
        self.initial_value = initial_value
        self.final_value = final_value
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Линейное увеличение в течение вармапа
            progress = self.current_step / self.warmup_steps
            return self.initial_value + progress * (self.final_value - self.initial_value)
        else:
            return self.final_value

def save_checkpoint(sae, optimizer, epoch, metrics, path):
    """Сохранение чекпоинта модели с метаданными"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': sae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Чекпоинт сохранён: {path}")

def train_sae(
    sae,
    activation_extractor,
    dataloader,
    num_epochs=1,
    lr=1e-3,
    sparsity_coeff=0.1,
    log_dir="./logs",
    device="cuda",
    checkpoint_dir="./checkpoints",
    save_every_epoch=True
):
    """
    Тренировочный цикл с логгированием через TensorBoard и сохранением чекпоинтов
    """
    # Создание директории для логов с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, timestamp)
    os.makedirs(log_path, exist_ok=True)

    # Инициализация логгера
    writer = SummaryWriter(log_dir=log_path)
    print(f"Логи сохраняются в: {log_path}")

    # Оптимизатор (только для энкодера и биаса)
    optimizer = optim.Adam([
        {'params': sae.encoder.parameters()},
        {'params': sae.encoder_bias}
    ], lr=lr)

    # Scheduler для разреженности
    scheduler = SAEScheduler(
        initial_value=1e-6,      # Начинаем с почти нулевой разреженности
        final_value=sparsity_coeff,
        warmup_steps=12000        # Дольше "разогреваем" коэффициент
    )

    # Обучение
    global_step = 0
    sae.train()

    best_ev = -float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        epoch_ev = 0.0

        progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device)

            # Извлечение активаций CLIP со слоя 11
            with torch.no_grad():
                activations = activation_extractor.extract_activations(images)

            # Прямой проход SAE
            x_hat, latents = sae(activations)
            current_sparsity = scheduler.step()
            total_loss, metrics = combined_loss(x_hat, activations, latents, sparsity_coeff=0.0)

            # Обратное распространение
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            if global_step % 50 == 0:
                with torch.no_grad():
                    # Проверка нормы столбцов декодера
                    decoder_norm = sae.decoder.weight.norm(dim=0).mean().item()
                    if decoder_norm < 0.95 or decoder_norm > 1.05:
                        print(f"Декодер вышел из нормы: {decoder_norm:.4f} (ожидалось ~1.0)")

            # Накопление метрик
            epoch_loss += total_loss.item()
            epoch_mse += metrics["mse_loss"]
            epoch_l1 += metrics["l1_loss"]
            epoch_l0 += metrics["l0_approx"]
            epoch_ev += metrics["explained_variance"]

            # Логгирование каждые 50 шагов
            if global_step % 50 == 0:
                writer.add_scalar("loss/total", total_loss.item(), global_step)
                writer.add_scalar("loss/mse", metrics["mse_loss"], global_step)
                writer.add_scalar("loss/l1", metrics["l1_loss"], global_step)
                writer.add_scalar("metrics/l0", metrics["l0_approx"], global_step)
                writer.add_scalar("metrics/explained_variance", metrics["explained_variance"], global_step)
                writer.add_scalar("sparsity_coeff", current_sparsity, global_step)

            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "L0": f"{metrics['l0_approx']:.1f}",
                "EV": f"{metrics['explained_variance']:.3f}"
            })

            global_step += 1

        # Логгирование метрик эпохи
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_l1 = epoch_l1 / num_batches
        avg_l0 = epoch_l0 / num_batches
        avg_ev = epoch_ev / num_batches

        writer.add_scalar("epoch/total_loss", avg_loss, epoch)
        writer.add_scalar("epoch/mse_loss", avg_mse, epoch)
        writer.add_scalar("epoch/l1_loss", avg_l1, epoch)
        writer.add_scalar("epoch/l0", avg_l0, epoch)
        writer.add_scalar("epoch/explained_variance", avg_ev, epoch)

        print(f"\nЭпоха {epoch+1} завершена:")
        print(f"  Средний лосс: {avg_loss:.6f}")
        print(f"  MSE: {avg_mse:.6f} | L1: {avg_l1:.6f}")
        print(f"  L0: {avg_l0:.2f} (целевой: {sae.k})")
        print(f"  Explained Variance: {avg_ev:.4f}")

        # Сохранение чекпоинта
        if save_every_epoch:
            checkpoint_path = os.path.join(checkpoint_dir, f"sae_epoch_{epoch+1:02d}.pth")
            save_checkpoint(sae, optimizer, epoch+1, {
                'loss': avg_loss,
                'mse': avg_mse,
                'l1': avg_l1,
                'l0': avg_l0,
                'explained_variance': avg_ev
            }, checkpoint_path)

        # Отслеживание лучшей модели по explained variance
        if avg_ev > best_ev:
            best_ev = avg_ev
            best_epoch = epoch + 1
            best_checkpoint_path = os.path.join(checkpoint_dir, "sae_best.pth")
            save_checkpoint(sae, optimizer, epoch+1, {
                'loss': avg_loss,
                'mse': avg_mse,
                'l1': avg_l1,
                'l0': avg_l0,
                'explained_variance': avg_ev,
                'is_best': True
            }, best_checkpoint_path)

    writer.close()
    print(f"\nОбучение завершено. Лучшая эпоха: {best_epoch} (EV={best_ev:.4f})")
    print(f"Логи: {log_path}")

    return sae, log_path