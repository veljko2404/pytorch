import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import math
import re

IMG_W, IMG_H = 280, 70 # images width and height
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
MIN_LEN = 3 # minimum text length
MAX_LEN = 14 # maximum text length
ALLOWED = set(ALPHABET) # set of valid characters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sinusoidal_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
    """
    Returns positional encoding of shape [max_len, 1, d_model] (broadcastable over batch).
    Stored as a non-trainable buffer in model.
    """
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(1)  # [max_len, 1, d_model]

class CNNTransformerOCR(nn.Module):
    """
    OCR model: CNN feature extractor -> Transformer encoder -> CTC classifier
    Input: [B, 1, 70, 280] (grayscale already, or you can convert before)
    Output: [T, B, num_classes] - logits for CTC loss; num_classes = vocab + 1 blank
    """
    def __init__(self, num_classes: int, img_h: int = 70, img_w: int = 280):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1: [B,1,70,280] -> [B,64,35,140]
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: [B,64,35,140] -> [B,128,17,140]  (height / 2, width same)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 3: [B,128,17,140] -> [B,256,8,140]
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 4: [B,256,8,140] -> [B,512,4,140]
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Collapse remaining height (4 -> 1): [B,512,4,140] -> [B,512,1,140]
            nn.Conv2d(512, 512, kernel_size=(4, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # find the sequence length T (max_T)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w) # fake input used only to infer CNN output shape
            out = self.cnn(dummy)  # forward pass without gradient tracking [1, 512, 1, 140]
            self.max_T = out.shape[-1] # sequence length after CNN (140 in our example)

        # Positional encoding buffer: [max_T, 1, 512]
        self.register_buffer("pos_enc", sinusoidal_pos_enc(self.max_T, 512), persistent=False)

        # Transformer encoder with 4 layers of multi-head self-attention
        enc_layer = nn.TransformerEncoderLayer(
            d_model=512,          # embedding dimension
            nhead=8,              # number of attention heads (512 / 8 = 64 per head)
            dim_feedforward=2048, # hidden size of the feed-forward network
            dropout=0.1,          # dropout for regularization
            batch_first=False,    # input shape: (seq_len, batch_size, d_model)
            norm_first=True,      # stabilizes training
            activation="gelu",    # often better than ReLU here
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)

        # classifier - projects each sequence position to class logits
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)              # [B,512,1,140]
        f = f.squeeze(2)             # [B,512,140]
        f = f.permute(2, 0, 1)       # [140,B,512]

        T = f.size(0)
        f = f + self.pos_enc[:T]     # [140,B,512] + [140,1,512] - adding positional encoding

        y = self.transformer(f)      # [140,B,512]
        return self.classifier(y)  # [140,B,num_classes]

class OCRDataset(Dataset):
    """
    Loads images and their text labels from a directory and CSV file.
    Filters out samples with missing images or invalid label lengths.
    """
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
            # Skip samples with text outside allowed length range
            if not (MIN_LEN <= len(tx_n) <= MAX_LEN):
                bad += 1
                continue
            # Skip samples whose image file does not exist
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

        # Load image, convert to grayscale and resize if needed
        img = Image.open(p).convert("L")
        if img.size != (IMG_W, IMG_H):
            img = img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)

        # Normalize pixel values to [0, 1] and add channel dimension
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # [H,W]
        x = x.unsqueeze(0)  # [1,H,W]
        y = encode_text(text)
        return x, y, text

def collate_fn(batch):
    """
    Custom collate function that merges variable-length label sequences
    into a single flat tensor, as required by PyTorch's CTCLoss
    xs      — stacked image tensors [B, 1, H, W]
    y_cat   — concatenated label indices [sum of label lengths]
    y_lens  — individual label lengths [B]
    texts   — list of ground-truth strings
    """
    xs, ys, texts = zip(*batch)
    xs = torch.stack(xs, dim=0)

    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    y_cat = torch.cat(ys, dim=0)

    return xs, y_cat, y_lens, list(texts)

def normalize_text(s: str) -> str:
    # strips, collapses whitespace, and removes characters outside the alphabet
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = "".join(ch for ch in s if ch in ALLOWED)
    return s

# Character <-> index mappings (index 0 is reserved for CTC blank)
char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
idx2char = {i + 1: c for i, c in enumerate(ALPHABET)}

def encode_text(text: str) -> torch.Tensor:
    # converts a text string to a tensor of character indices
    text = normalize_text(text)
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

def ctc_greedy_decode(logits: torch.Tensor) -> list[str]:
    """
    Greedy CTC decoding: takes argmax at each timestep,
    collapses repeated characters and removes blank tokens (index 0)
    """
    pred = logits.argmax(dim=-1)  # [T, B]
    pred = pred.detach().cpu().numpy()

    out = []
    for b in range(pred.shape[1]):
        seq = pred[:, b].tolist()
        collapsed = []
        prev = None
        for p in seq:
            if p != prev:
                collapsed.append(p)
            prev = p
        collapsed = [p for p in collapsed if p != 0]
        out.append("".join(idx2char.get(p, "") for p in collapsed))
    return out

def cer(pred: str, gt: str) -> float:
    """
    Character Error Rate (CER): edit distance between prediction and ground truth,
    normalized by the length of the ground truth string.
    """
    a, b = pred, gt
    if len(b) == 0:
        return 0.0 if len(a) == 0 else 1.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(
                dp[j] + 1,   # deletion
                dp[j-1] + 1, # insertion
                prev + cost  # substitution
            )
            prev = cur
    return dp[-1] / max(1, len(b))