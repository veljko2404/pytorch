import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import math
import re

IMG_W, IMG_H = 280, 70
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
MIN_LEN = 3
MAX_LEN = 14
ALLOWED = set(ALPHABET)

def sinusoidal_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
    """
    Returns positional encoding of shape [max_len, 1, d_model] (broadcastable over batch).
    Stored as a non-trainable buffer.
    """
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(1)  # [max_len, 1, d_model]

class CRNN(nn.Module):
    """
    CNN -> sequence (width as time) -> TransformerEncoder -> per-timestep logits for CTC.
    Input expected: [B, 1, 70, 280] (grayscale already, or you can convert before).
    Output: [T, B, num_classes]  where num_classes should be len(vocab) + 1 (CTC blank).
    """

    def __init__(
        self,
        num_classes: int,
        img_h: int = 70,
        img_w: int = 280,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        use_groupnorm: bool = False,
    ):
        super().__init__()

        def Norm(c: int):
            return nn.GroupNorm(32, c) if use_groupnorm else nn.BatchNorm2d(c)

        self.cnn = nn.Sequential(
            # [B,1,70,280]
            nn.Conv2d(1, 64, 3, padding=1), Norm(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> [B,64,35,140]

            nn.Conv2d(64, 128, 3, padding=1), Norm(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> [B,128,17,140]  (KEEP WIDTH)

            nn.Conv2d(128, 256, 3, padding=1), Norm(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), Norm(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> [B,256,8,140]

            nn.Conv2d(256, 512, 3, padding=1), Norm(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), Norm(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> [B,512,4,140]

            # Collapse height 4 -> 1, keep width (time) = 140
            nn.Conv2d(512, d_model, kernel_size=(4, 1), padding=0),
            Norm(d_model),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            out = self.cnn(dummy)  # [1, d_model, 1, T]
            self.max_T = out.shape[-1]

        # Positional encoding buffer: [max_T, 1, d_model]
        self.register_buffer("pos_enc", sinusoidal_pos_enc(self.max_T, d_model), persistent=False)

        # ---- Transformer encoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=False,
            norm_first=True,         # generally stabilizes training
            activation="gelu",       # often better than ReLU here
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ---- Classifier ----
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)              # [B,d_model,1,T]
        f = f.squeeze(2)             # [B,d_model,T]
        f = f.permute(2, 0, 1)       # [T,B,d_model]

        T = f.size(0)
        f = f + self.pos_enc[:T]     # [T,B,d_model] + [T,1,d_model]

        y = self.transformer(f)      # [T,B,d_model]
        logits = self.classifier(y)  # [T,B,num_classes]
        return logits

class OCRDataset(Dataset):
    def __init__(self, img_dir: Path, csv_path: Path):
        self.img_dir = Path(img_dir)
        df = pd.read_csv(csv_path)

        if not {"filename", "text"}.issubset(df.columns):
            raise ValueError("CSV must have columns: filename,text")

        samples = []
        missing = 0
        bad = 0
        for fn, tx in zip(df["filename"].astype(str), df["text"].astype(str)):
            tx_n = normalize_text(tx)
            if not (MIN_LEN <= len(tx_n) <= MAX_LEN):
                bad += 1
                continue
            p = self.img_dir / fn
            if not p.exists():
                missing += 1
                continue
            samples.append((fn, tx_n))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, text = self.samples[idx]
        p = self.img_dir / fn
        img = Image.open(p).convert("L")
        if img.size != (IMG_W, IMG_H):
            img = img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)

        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # [H,W]
        x = x.unsqueeze(0)  # [1,H,W]
        y = encode_text(text)
        return x, y, text

def collate_fn(batch):
    xs, ys, texts = zip(*batch)
    xs = torch.stack(xs, dim=0)

    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    y_cat = torch.cat(ys, dim=0)

    return xs, y_cat, y_lens, list(texts)

def normalize_text(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = "".join(ch for ch in s if ch in ALLOWED)
    return s

char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}

def encode_text(text: str) -> torch.Tensor:
    text = normalize_text(text)
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)