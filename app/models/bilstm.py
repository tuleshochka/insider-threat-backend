"""Bi-LSTM model for sequence classification of user events."""
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_MAP = {
    "<PAD>": 0, "<UNK>": 1, "LOGON": 2, "LOGOFF": 3, "USB_CONNECT": 4, 
    "USB_DISCONNECT": 5, "FILE_OPEN": 6, "FILE_WRITE": 7, "FILE_COPY": 8, 
    "FILE_DELETE": 9, "FILE_OPEN_USB": 10, "FILE_WRITE_USB": 11, 
    "FILE_COPY_USB": 12, "FILE_DELETE_USB": 13, "EMAIL_SEND_INT": 14, 
    "EMAIL_SEND_EXT": 15, "HTTP_BROWSE": 16, "HTTP_UPLOAD": 17, "HTTP_DOWNLOAD": 18
}

INV_TOKEN_MAP = {v: k for k, v in TOKEN_MAP.items()}
VOCAB_SIZE = len(TOKEN_MAP)
MAX_SEQ_LEN = 200

class BiLSTMClassifier(nn.Module):
    """
    Двунаправленная LSTM для классификации последовательностей действий пользователя.
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 64, 
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Возвращает логиты формы (batch,).
        """
        # x shape: (batch, seq_len)
        mask = (x != 0).float()  # (batch, seq_len) - маска для паддинга
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim*2)

        # Attention pooling
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)

        logits = self.classifier(context)  # (batch, 1)
        return logits.squeeze(-1)

    def get_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает кортеж (logits, attn_weights) для анализа интерпретируемости.
        """
        mask = (x != 0).float()
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)

        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        logits = self.classifier(context)
        return logits.squeeze(-1), attn_weights


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с сильным дисбалансом классов.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Вес для позитивного класса (1)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Взвешивание классов: alpha для target=1, 1-alpha для target=0
        alpha_t = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()


def map_event_to_token(event_type: str, details: dict) -> str:
    """
    Отображает тип события и его атрибуты из БД в строковый токен действия.
    """
    event_type = event_type.lower()
    details = details or {}
    
    if event_type == "logon":
        activity = str(details.get("activity", "")).strip().lower()
        return "LOGON" if "logon" in activity else "LOGOFF"
        
    elif event_type == "device":
        activity = str(details.get("activity", "")).strip().lower()
        return "USB_CONNECT" if "connect" in activity else "USB_DISCONNECT"
        
    elif event_type == "file":
        activity = str(details.get("activity", "")).strip().lower()
        # В CERT файловые события содержат признак to_removable_media / from_removable_media
        is_usb = (
            str(details.get("to_removable_media", "")).lower() == "true" or
            str(details.get("from_removable_media", "")).lower() == "true" or
            details.get("is_usb", False)
        )
        if "open" in activity:
            return "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
        elif "write" in activity:
            return "FILE_WRITE_USB" if is_usb else "FILE_WRITE"
        elif "copy" in activity:
            return "FILE_COPY_USB" if is_usb else "FILE_COPY"
        elif "delete" in activity:
            return "FILE_DELETE_USB" if is_usb else "FILE_DELETE"
        else:
            return "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
            
    elif event_type == "email":
        to_field = str(details.get("to", ""))
        # Если хотя бы один получатель не содержит домен организации, считаем внешним письмом
        has_external = any(
            r.strip() and "@dtaa.com" not in r 
            for r in to_field.split(";")
        )
        return "EMAIL_SEND_EXT" if has_external else "EMAIL_SEND_INT"
        
    elif event_type == "http":
        return "HTTP_BROWSE"
        
    return "<UNK>"
