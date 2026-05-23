"""PyTorch autoencoder model for anomaly detection."""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Fully-connected autoencoder with configurable hidden dimension.

    Архитектура:
        Вход (n) → Dense(//2, ReLU, Dropout) → Dense(n, Sigmoid)

    Зачем: классический undercomplete autoencoder с бутылочным горлышком.
    Принуждая данные проходить через слой меньшей размерности, модель учится
    восстанавливать только самые значимые признаки, отбрасывая шум.
    Это ключевое свойство для детектирования аномалий — аномальный образец
    будет восстановлен с большой ошибкой, поскольку его признаки не похожи
    на типичные комбинации, выученные на нормальных данных.

    Почему полносвязный: признаки табличные, без пространственной или
    временной структуры — свёрточные или рекуррентные слои избыточны.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        # Если размер скрытого слоя не указан явно — берём половину от входного,
        # но не менее 4, чтобы бутылочное горлышко не было слишком узким.
        hidden = hidden_dim if hidden_dim is not None else max(input_dim // 2, 4)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),   # Сжатие: input_dim → hidden
            nn.ReLU(),                      # Нелинейность: позволяет модели
                                            #   выучить сложные зависимости.
                                            #   ReLU выбран из-за устойчивости
                                            #   к затуханию градиента и малой
                                            #   вычислительной стоимости.
            nn.Dropout(0.2),                # Регуляризация: 20% нейронов
                                            #   обнуляется случайно на каждом
                                            #   шаге, чтобы сеть не запоминала
                                            #   отдельные признаки, а обобщала.
                                            #   0.2 — лёгкий dropout, достаточный
                                            #   для умеренного регуляризующего
                                            #   эффекта без потери ёмкости.
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, input_dim),   # Восстановление: hidden → input_dim
            nn.Sigmoid(),                   # Выход в [0, 1]: предполагается,
                                            #   что входные признаки нормализованы
                                            #   MinMax-масштабированием. Sigmoid
                                            #   гарантирует, что восстановленные
                                            #   значения лежат в том же диапазоне,
                                            #   и MSE-ошибка осмысленна.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns reconstructed input."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper — returns reconstruction."""
        return self.forward(x)


def train_autoencoder(
    model: Autoencoder,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 7,
    device: str | None = None,
) -> dict:
    """Train the autoencoder and return training history.

    Использует MSE-функцию потерь + оптимизатор Adam.
    Ранняя остановка, если потери на валидации не снижаются patience эпох подряд.

    Почему MSE (среднеквадратичная ошибка): аномалии порождают большие ошибки
    восстановления, а MSE квадратично штрафует за большие отклонения — это
    усиливает контраст между нормой и аномалией.

    Почему Adam: адаптивная скорость обучения, устойчив к выбору
    гиперпараметров, хорошо работает «из коробки» для задач небольшой
    размерности.

    Почему patience=7: даёт модели возможность «перетерпеть» локальное плато
    без улучшения, но не настолько большое, чтобы тратить время на
    переобучение.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # DataLoader может возвращать кортеж (данные, метки) —
            # если так, берём только первый элемент.
            batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()   # Обнуляем градиенты перед шагом
            loss.backward()         # Обратное распространение
            optimizer.step()        # Обновление весов
            train_loss += loss.item() * batch.size(0)

        # Средняя потеря на одном образце за эпоху
        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ── validation ──
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                    batch = batch.to(device)
                    recon = model(batch)
                    val_loss += criterion(recon, batch).item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    # Ранняя остановка: если валидационная потеря не падает
                    # patience эпох подряд — прекращаем обучение, чтобы
                    # избежать переобучения и сэкономить время.
                    break
        else:
            # Если валидационный набор не передан — используем тренировочную
            # потерю для ранней остановки. Запас в 5 эпох перед стартом
            # нужен, чтобы дать модели возможность начать обучаться.
            if epoch > 5 and train_loss >= history["train_loss"][-2]:
                stale += 1
                if stale >= patience:
                    break
            else:
                stale = 0

    model.to("cpu")  # Возвращаем на CPU для инференса на ноутбуке
    return history


def compute_anomaly_scores(
    model: Autoencoder,
    loader: torch.utils.data.DataLoader,
    device: str | None = None,
) -> list[float]:
    """Compute MSE reconstruction error for each sample.

    Чем выше ошибка восстановления, тем более аномальным считается образец.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch = batch.to(device)
            recon = model(batch)
            # reduction="none" — считаем MSE для каждого признака отдельно,
            # затем усредняем по всем признакам, получая одну оценку на образец.
            loss = nn.functional.mse_loss(recon, batch, reduction="none")
            per_sample = loss.mean(dim=1).cpu().tolist()
            scores.extend(per_sample)
    model.to("cpu")
    return scores
