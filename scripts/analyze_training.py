"""
Анализ метрик обучения SAE для отчёта (пункт 4 задания)
Генерирует таблицу в формате Markdown с ключевыми метриками
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(logdir):
    """Извлечение скаляров из логов TensorBoard (работает с прямым путём к запуску)"""
    # Проверяем, есть ли файлы событий в указанной директории
    event_files = [f for f in os.listdir(logdir) if f.startswith("events.out.tfevents")]

    if not event_files:
        # Если нет — ищем последнюю поддиректорию
        runs = [d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
        if not runs:
            raise FileNotFoundError(f"Не найдены логи в {logdir}")
        latest_run = sorted(runs)[-1]
        logdir = os.path.join(logdir, latest_run)
        event_files = [f for f in os.listdir(logdir) if f.startswith("events.out.tfevents")]
        if not event_files:
            raise FileNotFoundError(f"Не найдены файлы событий в {logdir}")

    # Загружаем первый найденный файл событий
    ea = event_accumulator.EventAccumulator(os.path.join(logdir, event_files[0]))
    ea.Reload()

    # Извлечение метрик последней эпохи
    scalars = {}
    for tag in ['epoch/total_loss', 'epoch/mse_loss', 'epoch/l1_loss', 'epoch/l0', 'epoch/explained_variance']:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            scalars[tag.replace('epoch/', '')] = events[-1].value

    return scalars, os.path.basename(logdir)


def load_sae_config(config_path="configs/sae_config.yaml"):
    """Загрузка конфигурации SAE для расчёта dict_size"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['sae']['d_model'] * cfg['sae']['expansion_factor']


def generate_metrics_table(metrics, dict_size, output_path):
    """Генерация таблицы метрик в формате Markdown"""
    table = f"""## Метрики обучения SAE

| Метрика | Значение | Описание |
|---------|----------|----------|
| **dict_size** | {dict_size:,} | Размер словаря фичей (`d_model × expansion_factor`) |
| **L0** | {metrics['l0']:.1f} | Среднее число активных фичей на изображение |
| **Explained Variance Ratio** | {metrics['explained_variance']:.4f} | Доля дисперсии активаций, объяснённая реконструкцией |
| **MSE Loss** | {metrics['mse_loss']:.6f} | Ошибка реконструкции активаций |
| **L1 Loss** | {metrics['l1_loss']:.6f} | Сумма абсолютных активаций (мера разреженности) |
| **Total Loss** | {metrics['total_loss']:.6f} | Комбинированный лосс: `MSE + λ·L1` |

*Обучение: 20,000 изображений ImageNet validation, 5 эпох, `batch_size=64`, `expansion_factor=32`*
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(table)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logs/sae_training",
                       help="Директория с логами обучения")
    parser.add_argument("--sae_config", type=str, default="configs/sae_config.yaml",
                       help="Путь к конфигурации SAE")
    parser.add_argument("--output", type=str, default="./data/metrics_table.md",
                       help="Путь для сохранения таблицы метрик")
    args = parser.parse_args()

    # Анализ логов
    metrics, run_id = parse_tensorboard(args.logdir)

    # Расчёт dict_size
    dict_size = load_sae_config(args.sae_config)

    # Генерация таблицы
    table = generate_metrics_table(metrics, dict_size, args.output)

    print(table)
    print(f"\n Таблица метрик сохранена: {args.output}")


if __name__ == "__main__":
    main()