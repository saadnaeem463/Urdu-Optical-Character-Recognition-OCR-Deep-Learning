# ocr_lib.py - preprocessing, line splitting, model and decoding.
# Shared by the training notebook and the demo app, so training and inference can never drift apart.
import numpy as np
import torch, torch.nn as nn
from PIL import Image

IMG_H = 96            # every text line is resized to this height (must be a multiple of 32)
DOWN = 4              # the CNN shrinks the width by 4 -> one CTC time step per 4 pixels
TRAIN_MAX_W = 2048    # width cap while training
INFER_MAX_W = 4096    # width cap at inference (conv + LSTM accept longer lines than seen in training)


# ---------------------------------------------------------------- preprocessing
def normalize_gray(img):
    """PIL image -> uint8 gray array with DARK text on a LIGHT background and stretched contrast."""
    g = np.array(img.convert("L"), dtype=np.uint8)
    if g.size == 0:
        return g
    if g.mean() < 110:                              # light text on a dark background -> flip it
        g = 255 - g
    lo, hi = int(g.min()), int(np.percentile(g, 90))
    if hi - lo >= 30:                               # stretch contrast (yellowed paper, faded scans)
        g = np.clip((g.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return g


def ink_mask(g):
    """True where a pixel looks like text (darker than the midpoint between ink and background)."""
    lo, hi = int(g.min()), int(np.percentile(g, 90))
    if hi - lo < 30:
        return np.zeros(g.shape, dtype=bool)
    return g < (lo + hi) / 2.0


def auto_crop(g, pad_frac=0.12):
    """Cut away empty margins around the text, keeping a small border (pad_frac of the text height)."""
    m = ink_mask(g)
    rp, cp = m.sum(1), m.sum(0)
    if rp.max() == 0:
        return g
    rows = np.flatnonzero(rp >= max(2, 0.03 * rp.max()))
    cols = np.flatnonzero(cp >= max(2, 0.03 * cp.max()))
    if len(rows) == 0 or len(cols) == 0:
        return g
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    pad = int(round(pad_frac * (y1 - y0)))
    return g[max(0, y0 - pad):min(g.shape[0], y1 + pad), max(0, x0 - pad):min(g.shape[1], x1 + pad)]


def _runs(flags):
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append([start, i]); start = None
    if start is not None:
        runs.append([start, len(flags)])
    return runs


def _ref_height(runs):
    hs = np.array([e - s for s, e in runs], dtype=np.float32)
    return float(np.median(hs[hs >= 0.5 * hs.max()]))


def _dense_split(g, m):
    """Lines that touch or overlap (no empty rows between them): cut at the valleys of the smoothed row profile.
    Returns a list of line crops, or None if the image does not look like several similar lines."""
    prof = m.sum(1).astype(np.float32)
    H = len(prof)
    k = max(3, (H // 80) | 1)
    sm = np.convolve(prof, np.ones(k, np.float32) / k, mode="same")
    mx = float(sm.max())
    if mx <= 0:
        return None
    for frac in (0.05, 0.10, 0.15, 0.22):               # try a stricter and stricter notion of "gap"
        runs = _runs(sm >= frac * mx)
        if len(runs) < 2:
            continue
        hs = np.array([e - s for s, e in runs], dtype=np.float32)
        med = float(np.median(hs))
        if hs.min() < 0.45 * med or hs.max() > 1.7 * med:   # unequal pieces = not a stack of similar lines
            continue
        if any(s1 - e0 < max(3.0, 0.1 * med) for (_, e0), (s1, _) in zip(runs[:-1], runs[1:])):
            continue                                    # gaps this thin are just a gap inside one line
        cuts = [0]
        for (s0, e0), (s1, e1) in zip(runs[:-1], runs[1:]):
            cuts.append(e0 + int(np.argmin(sm[e0:s1])) if s1 > e0 else e0)   # lowest point inside the gap
        cuts.append(H)
        lines = [auto_crop(g[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 4]
        hc = np.array([l.shape[0] for l in lines], dtype=np.float32)
        if len(lines) >= 2 and hc.min() >= 0.4 * np.median(hc) and hc.max() <= 1.8 * np.median(hc):
            return lines                                # the pieces must still look like similar lines after cropping
    return None


def split_lines(g):
    """Split a page / multi-line image into single-line crops, top to bottom (row projection profile)."""
    m = ink_mask(g)
    prof = m.sum(1).astype(np.float32)
    if prof.max() == 0:
        return [g]
    runs = _runs(prof >= max(2.0, 0.01 * prof.max()))
    if len(runs) <= 1:                                  # no empty row between lines: maybe tightly set text
        dense = _dense_split(g, m)
        return dense if dense else [auto_crop(g)]
    ref = _ref_height(runs)
    big = [list(r) for r in runs if r[1] - r[0] >= 0.5 * ref]       # real lines
    if not big:
        return [auto_crop(g)]
    for s0, e0 in (r for r in runs if r[1] - r[0] < 0.5 * ref):     # dots / diacritics / specks: attach to the nearest line
        dist = [max(b[0] - e0, s0 - b[1], 0) for b in big]
        k = int(np.argmin(dist))
        if dist[k] <= 0.5 * ref:
            big[k][0], big[k][1] = min(big[k][0], s0), max(big[k][1], e0)
    ref = _ref_height(big)
    lines = []
    for s0, e0 in big:
        p = int(0.15 * (e0 - s0))
        piece = g[max(0, s0 - p):min(g.shape[0], e0 + p)]
        if (e0 - s0) > 1.8 * ref:                       # one block that is really several touching lines
            sub = _dense_split(piece, ink_mask(piece))
            if sub:
                lines.extend(sub)
                continue
        lines.append(auto_crop(piece))
    return lines


def resize_line(g, max_w, width_factor=1.0):
    """Resize to height IMG_H keeping the aspect ratio (width is a multiple of DOWN)."""
    h, w = g.shape
    new_w = int(round(w * IMG_H / h * width_factor))
    new_w = max(DOWN * 4, min(max_w, new_w))
    new_w -= new_w % DOWN
    return np.array(Image.fromarray(g).resize((new_w, IMG_H), Image.BILINEAR))


def to_tensor(g):
    """uint8 (H, W) -> float tensor (1, H, W), white = 1. Mirrored because Urdu is written right to left."""
    g = np.ascontiguousarray(g[:, ::-1], dtype=np.uint8)
    return torch.from_numpy(g).float().div(255).unsqueeze(0)


# ---------------------------------------------------------------- model
def _cb(i, o):
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, num_classes, img_h=IMG_H, rnn_hidden=256, rnn_layers=3, dropout=0.25):
        super().__init__()
        layers = (_cb(1, 64) + [nn.MaxPool2d((2, 2))] +
                  _cb(64, 128) + [nn.MaxPool2d((2, 2))] +
                  _cb(128, 256) + _cb(256, 256) + [nn.MaxPool2d((2, 1))] +
                  _cb(256, 512) + _cb(512, 512) + [nn.MaxPool2d((2, 1))] +
                  _cb(512, 512) + [nn.MaxPool2d((2, 1))])
        self.cnn = nn.Sequential(*layers)               # height /32, width /4
        feat = 512 * (img_h // 32)                      # 96 -> 3 rows x 512 channels = 1536 numbers per slice
        self.proj = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=rnn_layers, bidirectional=True,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, x):                               # x: (B, 1, H, W)
        x = self.cnn(x)                                 # (B, 512, H/32, W/4)
        b, c, h, w = x.shape
        x = x.reshape(b, c * h, w).permute(0, 2, 1)     # (B, slices, features)
        x = self.proj(x)
        x, _ = self.rnn(x)
        return self.fc(x)                               # (B, slices, num_classes)


# ---------------------------------------------------------------- decoding / prediction
def ids_to_text(ids, length, idx2char):
    """CTC greedy decode of one sequence: merge repeats, drop blanks (0). Only the first `length` slices are real."""
    text, prev = [], 0
    for k in ids[:length]:
        if k != prev and k != 0:
            text.append(idx2char[k])
        prev = k
    return "".join(text)


def decode_batch(out, lens, idx2char):
    best = out.argmax(2).cpu().tolist()
    return [ids_to_text(b, int(l), idx2char) for b, l in zip(best, lens)]


@torch.no_grad()
def predict_arrays(model, arrays, idx2char, device="cpu", max_w=INFER_MAX_W):
    model.eval()
    texts = []
    for g in arrays:
        x = to_tensor(resize_line(g, max_w)).unsqueeze(0).to(device)     # (1, 1, H, W)
        out = model(x)
        texts.append(decode_batch(out.float(), [x.shape[3] // DOWN], idx2char)[0])
    return texts


def read_image(img, model, idx2char, device="cpu", split=True):
    """PIL image (one line, an uncropped line, or a multi-line page) -> list of recognized lines."""
    g = normalize_gray(img)
    lines = split_lines(g) if split else [auto_crop(g)]
    lines = [l for l in lines if l.size and min(l.shape) >= 4]
    return predict_arrays(model, lines, idx2char, device)