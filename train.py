"""
Training script for MyChatAI.
Builds vocab, trains the GPT, saves model + tokenizer.
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from model import GPT
from data import CONVERSATIONS, ConversationDataset, collate_fn

# ── Config ─────────────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'weights')
MODEL_PATH = os.path.join(SAVE_DIR, 'model.pt')
VOCAB_PATH = os.path.join(SAVE_DIR, 'vocab.json')

EPOCHS = 80           # more epochs = smarter but slower
BATCH_SIZE = 16
LR = 3e-4
WARMUP_STEPS = 200
GRAD_CLIP = 1.0

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 1024
MAX_SEQ = 256
DROPOUT = 0.1

MAX_VOCAB = 6000


def cosine_lr(step, warmup, total):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'   # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = 'cuda'
    print(f"[train] device: {device}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("[train] building vocabulary …")
    tok = Tokenizer()
    all_texts = [u for u, _ in CONVERSATIONS] + [b for _, b in CONVERSATIONS]
    tok.build(all_texts, max_vocab=MAX_VOCAB)
    tok.save(VOCAB_PATH)

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = ConversationDataset(tok, max_len=MAX_SEQ)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, drop_last=False)
    print(f"[train] {len(dataset)} samples  |  {len(loader)} batches/epoch")

    # ── Model ──────────────────────────────────────────────────────────────
    model = GPT(
        vocab_size=len(tok),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq=MAX_SEQ,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_lr(s, WARMUP_STEPS, max(total_steps, WARMUP_STEPS + 1))
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n[train] starting training — {EPOCHS} epochs\n")
    step = 0
    best_loss = float('inf')

    bar_width = 30

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # mask padding in targets (-100 is ignored by cross_entropy)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        pct = epoch / EPOCHS
        filled = int(bar_width * pct)
        bar = '█' * filled + '░' * (bar_width - filled)

        print(f"  epoch {epoch:3d}/{EPOCHS}  [{bar}]  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}  {elapsed:.1f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)

        # quick sample every 20 epochs
        if epoch % 20 == 0:
            _sample(model, tok, device, "how are you")

    print(f"\n[train] done! best loss: {best_loss:.4f}")
    print(f"[train] model saved → {MODEL_PATH}")
    print(f"[train] vocab saved → {VOCAB_PATH}")


def _sample(model, tok, device, prompt):
    model.eval()
    with torch.no_grad():
        ids = tok.encode_prompt(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=60, temperature=0.8, top_p=0.9)
        tokens = out[0].tolist()[len(ids):]
        reply = tok.decode(tokens)
    print(f"  [sample] '{prompt}' → '{reply}'")
    model.train()


if __name__ == '__main__':
    train()
