"""
Word-level tokenizer built from scratch.
Splits on whitespace + punctuation, learns vocabulary from corpus.
"""

import re
import json
import os


SPECIAL = ['<PAD>', '<EOS>', '<UNK>', '<BOT>', '<EOT>']
PAD_ID, EOS_ID, UNK_ID, BOT_ID, EOT_ID = 0, 1, 2, 3, 4


def _split(text):
    """Lowercase and split text into tokens (words + punctuation)."""
    text = text.lower().strip()
    # separate punctuation from words
    tokens = re.findall(r"[a-z0-9']+|[^a-z0-9\s]", text)
    return tokens


class Tokenizer:
    def __init__(self):
        self.vocab = {}        # token -> id
        self.inv_vocab = {}    # id -> token

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, texts, max_vocab=8000):
        from collections import Counter
        counts = Counter()
        for text in texts:
            for tok in _split(text):
                counts[tok] += 1

        # keep the most common words
        common = [w for w, _ in counts.most_common(max_vocab - len(SPECIAL))]

        self.vocab = {s: i for i, s in enumerate(SPECIAL)}
        for w in common:
            self.vocab[w] = len(self.vocab)

        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        print(f"[tokenizer] vocab size: {len(self.vocab)}")

    # ── Encode / decode ────────────────────────────────────────────────────

    def encode(self, text, add_eos=False):
        ids = [self.vocab.get(t, UNK_ID) for t in _split(text)]
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOT_ID, EOT_ID):
                continue
            tokens.append(self.inv_vocab.get(i, '<UNK>'))

        # rejoin — add space between words but not before punctuation
        out = ''
        for tok in tokens:
            if tok in (',', '.', '!', '?', ':', ';', "'", ')'):
                out += tok
            elif out.endswith("'") or out.endswith('('):
                out += tok
            else:
                out += (' ' if out else '') + tok
        return out.strip()

    # ── Conversation helpers ───────────────────────────────────────────────

    def encode_prompt(self, user_text):
        """Encode a user turn + BOT token ready for generation."""
        return [BOT_ID] + self.encode(user_text) + [EOT_ID, BOT_ID]

    def encode_pair(self, user_text, bot_text):
        """Encode a full (user, bot) pair as a training sequence."""
        return (
            [BOT_ID]
            + self.encode(user_text)
            + [EOT_ID, BOT_ID]
            + self.encode(bot_text)
            + [EOS_ID]
        )

    # ── Save / load ────────────────────────────────────────────────────────

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    def load(self, path):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)
