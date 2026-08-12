"""Train the history-aware sequence scorer (Gambit, Appendix A.1).

Unlike a Markovian MLP that scores each step in isolation, the sequence scorer
maps a *prefix* of per-step hidden states to a scalar quality score in [0, 1] by
attending over the full reasoning trajectory:

    h~_t = GELU(W_in · LN(h_t))
    z_1..z_T = SequenceTransformer(h~_1..h~_T)
    y^_t = sigmoid(w_out^T · LN(z_t))

Architecture (SequenceScorer):
    LayerNorm(input_dim) -> Linear(input_dim, d_model) -> GELU   # input projection
    N x SequenceTransformerLayer:
        Pre-norm multi-head causal self-attention with RoPE
        Pre-norm SwiGLU FFN (gate + up + down projections)
    LayerNorm(d_model) -> Linear(d_model, 1)                     # score head

The causal mask is essential: position t attends only to steps 1..t, so y^_t is
conditioned on the full reasoning history up to step t. This lets the scorer flag
globally inferior steps that look locally plausible.

Training objective — last-step BCE. Each trace is one sample: the full sequence
[h_1..h_T] is fed in, but loss is taken *only* at the last valid position T,
against the trace-level correctness label y. Only the final position contributes
a gradient, forcing attention to credit-assign over the whole trajectory rather
than relying on local features. This matches inference, where the engine reads
the logit at the final observed step.

Input: sharded .pt files produced by extract_hidden_states.py.

Usage:
    # d_model=256, 1 layer (~1.6M params)
    python gambit/train_scorer/train_sequence_scorer.py \\
      --train_dir  <hidden_states>/train \\
      --test_dir   <hidden_states>/test \\
      --d_model 256 --nhead 4 --num_layers 1 --dropout 0.1 --max_len 4096 \\
      --epochs 30 --batch_size 8 --lr 1e-4 --config_name seq_d256_1l

    # d_model=512, 2 layers
    python gambit/train_scorer/train_sequence_scorer.py \\
      --train_dir  <hidden_states>/train \\
      --test_dir   <hidden_states>/test \\
      --d_model 512 --nhead 8 --num_layers 2 --dropout 0.1 --max_len 4096 \\
      --epochs 30 --batch_size 4 --lr 5e-5 --config_name seq_d512_2l
"""

import os
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the history-aware sequence scorer (last-step BCE on full traces)"
    )
    # Data
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory with .pt hidden-state files (train split)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory with .pt hidden-state files (held-out test). '
                             'Optional; validation is always split from train_dir.')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of train_dir used for validation (default: 0.15)')
    parser.add_argument('--max_len', type=int, default=4096,
                        help='Maximum sequence length; longer traces are truncated '
                             '(default: 4096)')

    # Model
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer model dimension (default: 256)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of transformer layers (default: 1)')
    parser.add_argument('--dim_feedforward', type=int, default=0,
                        help='SwiGLU hidden dim; 0 = 4×d_model (default: 0)')
    parser.add_argument('--rope_base', type=int, default=10000,
                        help='RoPE base frequency. Default 10000 (LLM standard). '
                             'For step sequences capped at ~2K, base≈300-500 gives '
                             'better angular resolution (default: 10000)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size in traces (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='AdamW weight decay (default: 1e-5)')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['none', 'cosine'],
                        help='LR schedule: none / cosine (default)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Linear LR warmup epochs (default: 2)')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early-stopping patience (default: 8)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--l1_reg', type=float, default=0.0,
                        help='L1 regularization coefficient (default: 0.0, disabled)')
    parser.add_argument('--l2_reg', type=float, default=0.0,
                        help='L2 regularization coefficient added on top of AdamW weight_decay (default: 0.0)')

    # Checkpoint / output
    parser.add_argument('--config_name', type=str, default='sequence_default',
                        help='Name for checkpoint directory and plot labels')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a checkpoint to resume from')

    return parser.parse_args()


# ============================================================================
# Model building blocks
# ============================================================================

class RoPEAttention(nn.Module):
    """Multi-head causal self-attention with Rotary Position Embedding (RoPE).

    Position is encoded by rotating Q and K vectors with position-dependent
    angles, exactly as in DeepSeek/LLaMA.  No separate positional embedding
    table is needed.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0,
                 rope_base: int = 10000):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead     = nhead
        self.d_head    = d_model // nhead
        self.qkv       = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head)
        )
        self.register_buffer("inv_freq", inv_freq)   # [d_head / 2]

    def _rope_cos_sin(self, T: int, device: torch.device,
                      dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t     = torch.arange(T, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))   # [T, d_head/2]
        emb   = torch.cat([freqs, freqs], dim=-1)          # [T, d_head]
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, nh, T, d_head]

        # Apply RoPE to Q and K
        cos, sin = self._rope_cos_sin(T, x.device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, d_head]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        # Additive attention bias: causal + optional key padding
        attn_bias = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_bias = attn_bias.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        if key_padding_mask is not None:
            # [B, T] → [B, 1, 1, T]: mask out padding *key* positions for all queries
            pad_bias = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
            pad_bias = pad_bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_bias = attn_bias + pad_bias

        dp  = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: output = SiLU(gate(x)) ⊙ up(x), then down(x).

    Three weight matrices (gate, up, down) instead of the standard two.
    For parameter-count parity with a vanilla FFN of width W, set
    dim_feedforward ≈ 2W/3.
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.up   = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down = nn.Linear(dim_feedforward, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class SequenceTransformerLayer(nn.Module):
    """Pre-norm causal transformer layer: RoPE attention + SwiGLU FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0, rope_base: int = 10000):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = RoPEAttention(d_model, nhead, dropout, rope_base=rope_base)
        self.ffn   = SwiGLUFFN(d_model, dim_feedforward, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ============================================================================
# Model
# ============================================================================

class SequenceScorer(nn.Module):
    """History-aware sequence scorer: N-layer causal Transformer with RoPE + SwiGLU.

    Input:  x  [B, T, input_dim]  — sequence of per-step hidden states
            key_padding_mask  [B, T]  — True at padding positions
    Output: logits  [B, T, 1]  — unnormalised score at each step position
    """

    def __init__(
        self,
        input_dim: int = 4096,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 0,
        dropout: float = 0.1,
        max_len: int = 4096,
        num_layers: int = 1,
        rope_base: int = 10000,
    ):
        super().__init__()
        if dim_feedforward <= 0:
            dim_feedforward = d_model * 4

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )
        # No learnable positional embedding — position is encoded via RoPE
        self.max_len = max_len
        self.layers  = nn.ModuleList([
            SequenceTransformerLayer(d_model, nhead, dim_feedforward, dropout,
                                   rope_base=rope_base)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SequenceScorer: input_dim={input_dim}, d_model={d_model}, nhead={nhead}, "
              f"ffn={dim_feedforward}, dropout={dropout}, "
              f"num_layers={num_layers}, max_len={max_len}, rope_base={rope_base}")
        print(f"  Architecture: RoPE attention + SwiGLU FFN")
        print(f"  Total parameters: {n_params:,}")

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.norm(x)
        return self.head(x)   # [B, T, 1]


# ============================================================================
# Datasets
# ============================================================================

class TraceDataset(Dataset):
    """Full-trace dataset.

    Each item returns the full hidden-state sequence [T, D] and its label.
    No step sub-sampling — one trace = one sample.
    """

    def __init__(self, items: list, max_len: int = 4096):
        self.items = items
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        hs = item['select_hidden_states']
        T = min(hs.shape[0], self.max_len)
        hs = hs[:T].half().contiguous()   # store as float16; cast to float32 on GPU
        label = 1.0 if item.get('is_correct', False) else 0.0
        return hs, label, T


# ============================================================================
# Collate function
# ============================================================================

def collate_traces(batch):
    """Pad a batch of full traces."""
    max_T = max(item[2] for item in batch)
    D = batch[0][0].shape[1]
    B = len(batch)

    x    = torch.zeros(B, max_T, D, dtype=torch.float16)
    labs = torch.zeros(B)
    mask = torch.ones(B, max_T, dtype=torch.bool)

    for i, (hs, label, T) in enumerate(batch):
        x[i, :T] = hs
        labs[i] = label
        mask[i, :T] = False

    return x, labs, mask


# ============================================================================
# Data Loading
# ============================================================================

def load_items_from_dir(file_path: str) -> list:
    items = []
    if not os.path.exists(file_path):
        print(f"Path not found: {file_path}")
        return items
    filenames = sorted(f for f in os.listdir(file_path) if f.endswith('.pt'))
    for fname in tqdm(filenames, desc=f"Loading {file_path}"):
        try:
            data = torch.load(os.path.join(file_path, fname),
                              map_location='cpu', weights_only=False)
            if isinstance(data, list):
                items.extend(data)
        except Exception as e:
            print(f"  Error loading {fname}: {e}")
    return items


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray,
                     prefix: str = '') -> dict:
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5
    print(f"\n{prefix}Results:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}\n")
    return dict(auc=auc, acc=acc, f1=f1, precision=prec, recall=rec)


def run_validation_last(
    model: nn.Module,
    val_items: list,
    max_len: int,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate using last-step logit on full traces (mirrors ::last inference).

    Uses a small batch_size (default 4) to avoid OOM on long sequences.
    """
    model.eval()
    ds     = TraceDataset(val_items, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_traces, num_workers=0)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, labels, mask in loader:
            x, mask = x.to(DEVICE, dtype=torch.float32), mask.to(DEVICE)
            logits = model(x, key_padding_mask=mask).squeeze(-1)  # [B, T]
            valid  = (~mask)
            # 0-based index of last valid position per trace
            seq_lens  = valid.sum(dim=1).clamp(min=1) - 1
            last_logit = logits[torch.arange(len(seq_lens), device=logits.device), seq_lens]
            all_scores.append(torch.sigmoid(last_logit).cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ============================================================================
# Training Loop
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_items: list,
    num_pos: int,
    num_neg: int,
    args,
    input_dim: int,
    checkpoint_dir: str,
    resume_optimizer_state=None,
    resume_scheduler_state=None,
) -> tuple:
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(DEVICE)

    # Class-weighted BCE
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)
        print(f"Class distribution — pos: {num_pos}, neg: {num_neg}, "
              f"pos_weight: {pos_weight.item():.4f}")
    else:
        pos_weight = torch.tensor([1.0]).to(DEVICE)
        print("Warning: no positive samples found.")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            print(f"  Optimizer state restored (lr={args.lr})")
        except Exception as e:
            print(f"  Warning: could not restore optimizer state ({e})")

    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    if resume_scheduler_state is not None:
        try:
            scheduler.load_state_dict(resume_scheduler_state)
            print("  Scheduler state restored.")
        except Exception as e:
            print(f"  Warning: could not restore scheduler state ({e})")

    loss_curve = []
    best_val_score = float('-inf')
    best_model_state = None
    patience_counter = 0

    print(f"\nTraining '{args.config_name}' on {DEVICE}  "
          f"(epochs={args.epochs}, warmup={args.warmup_epochs}, schedule={args.lr_schedule})")
    print(f"Training traces per epoch: {len(train_loader.dataset):,}")
    print(f"Batches per epoch:         {len(train_loader):,}")
    print("-" * 60)

    for epoch in range(args.epochs):
        # LR warmup
        if epoch < args.warmup_epochs:
            lr_now = args.lr * (epoch + 1) / max(1, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_traces   = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for x, labels, mask in pbar:
            x      = x.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE)
            mask   = mask.to(DEVICE)
            B      = x.shape[0]

            optimizer.zero_grad()

            logits = model(x, key_padding_mask=mask).squeeze(-1)  # [B, T]

            # Loss only at the last valid (non-padding) position per trace.
            # Each trace contributes exactly one gradient signal with full causal context.
            seq_lens   = (~mask).sum(dim=1).clamp(min=1) - 1       # [B]
            last_logit = logits[torch.arange(B, device=logits.device), seq_lens]  # [B]
            loss = criterion(last_logit, labels)

            if args.l1_reg > 0.0 or args.l2_reg > 0.0:
                for param in model.parameters():
                    if param.requires_grad:
                        if args.l1_reg > 0.0:
                            loss = loss + args.l1_reg * param.abs().sum()
                        if args.l2_reg > 0.0:
                            loss = loss + args.l2_reg * param.pow(2).sum()

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * B
            n_traces   += B
            pbar.set_postfix(loss=f"{epoch_loss / n_traces:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        pbar.close()

        avg_train_loss = epoch_loss / max(n_traces, 1)
        loss_curve.append(avg_train_loss)

        # --- Validation (last-step AUC on full traces, mirrors ::last) ---
        val_scores, val_labels = run_validation_last(model, val_items, args.max_len)
        metrics = evaluate_metrics(
            val_labels.astype(int), val_scores,
            prefix=f"[{args.config_name}] Epoch {epoch+1} "
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{args.config_name}] Epoch [{epoch+1}/{args.epochs}]  "
              f"train_loss={avg_train_loss:.4f}  lr={current_lr:.2e}")

        # Scheduler step (after warmup)
        if epoch >= args.warmup_epochs and args.lr_schedule == 'cosine':
            scheduler.step()

        # --- Early stopping on last-step AUC ---
        current_score = 0.75 * metrics['auc'] + 0.25 * metrics['f1']
        if current_score > best_val_score + 1e-4:
            best_val_score = current_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            ckpt_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch{epoch+1}_auc{metrics['auc']:.4f}_f1{metrics['f1']:.4f}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_val_score,
                'metrics': metrics,
                'config_name': args.config_name,
                'loss_curve': loss_curve,
                'model_config': {
                    'type': 'sequence_transformer',
                    'input_dim': input_dim,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'dim_feedforward': (args.dim_feedforward if args.dim_feedforward > 0
                                        else args.d_model * 4),
                    'dropout': args.dropout,
                    'max_len': args.max_len,
                    'num_layers': args.num_layers,
                    'rope_base': args.rope_base,
                },
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")
            print(f"  -> New best: {current_score:.4f}  "
                  f"(AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nTraining complete. Final validation with best model...")
    val_scores, val_labels = run_validation_last(model, val_items, args.max_len)
    val_metrics = evaluate_metrics(
        val_labels.astype(int), val_scores,
        prefix=f"[{args.config_name}] Final Validation "
    )
    return loss_curve, val_metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_loss_curves(results_dict: dict, output_dir: str = './plots'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    for name, data in results_dict.items():
        if data.get('loss_curve'):
            plt.plot(data['loss_curve'], label=name, linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)')
    plt.title('Training Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    print(f"Loss curves saved to {output_dir}/loss_curves.png")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # ---- Load training items ----
    print(f"Loading training data from: {args.train_dir}")
    all_items = load_items_from_dir(args.train_dir)
    if not all_items:
        print("No training data found. Exiting.")
        return
    print(f"Total training traces: {len(all_items)}")

    # ---- Split train / val (and optionally use a separate test dir) ----
    if args.test_dir:
        print(f"Loading test data from: {args.test_dir}")
        test_items = load_items_from_dir(args.test_dir)
        print(f"Total test traces: {len(test_items)}")
    else:
        test_items = None

    n_val   = max(1, int(len(all_items) * args.val_split))
    n_train = len(all_items) - n_val
    rng_split = torch.Generator().manual_seed(RANDOM_STATE)
    train_idx, val_idx = random_split(range(len(all_items)), [n_train, n_val],
                                       generator=rng_split)
    train_items = [all_items[i] for i in train_idx]
    val_items   = [all_items[i] for i in val_idx]

    # Class distribution
    def class_dist(items):
        n_pos = sum(1 for x in items if x.get('is_correct'))
        return n_pos, len(items) - n_pos

    num_pos, num_neg = class_dist(train_items)
    v_pos, v_neg     = class_dist(val_items)
    print(f"Train: {len(train_items)} traces  |  Val: {len(val_items)} traces"
          + (f"  |  Test: {len(test_items)} traces" if test_items else ""))
    print(f"Train class dist: pos={num_pos}, neg={num_neg}")
    print(f"Val class dist:   pos={v_pos}, neg={v_neg}")

    lengths = [min(x['select_hidden_states'].shape[0], args.max_len) for x in train_items]
    print(f"Step lengths (capped at {args.max_len}) — "
          f"min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")

    input_dim = train_items[0]['select_hidden_states'].shape[-1]
    print(f"Input dimension: {input_dim}")

    # ---- Build training dataset & loader ----
    train_ds = TraceDataset(train_items, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_traces, num_workers=4, pin_memory=True,
        prefetch_factor=1,
    )

    # ---- Build model ----
    model = SequenceScorer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        num_layers=args.num_layers,
        rope_base=args.rope_base,
    )

    resume_optimizer_state = None
    resume_scheduler_state = None
    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            resume_optimizer_state = ckpt.get('optimizer_state_dict')
            resume_scheduler_state = ckpt.get('scheduler_state_dict')
            prev = ckpt.get('best_score')
            if prev is not None:
                print(f"  Previous best score: {prev:.4f} (epoch {ckpt.get('epoch')})")
        else:
            model.load_state_dict(ckpt)
        print("  Weights loaded.")

    # ---- Train ----
    checkpoint_dir = f'./checkpoints/{args.config_name}'
    loss_curve, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_items=val_items,
        num_pos=num_pos,   # trace-level counts
        num_neg=num_neg,
        args=args,
        input_dim=input_dim,
        checkpoint_dir=checkpoint_dir,
        resume_optimizer_state=resume_optimizer_state,
        resume_scheduler_state=resume_scheduler_state,
    )

    # ---- Test set evaluation ----
    if test_items:
        print("\n" + "=" * 60)
        print("Evaluating on held-out test set...")
        test_scores, test_labels = run_validation_last(model, test_items, args.max_len)
        test_metrics = evaluate_metrics(
            test_labels.astype(int), test_scores,
            prefix=f"[{args.config_name}] Test Set "
        )
    else:
        test_metrics = None

    # ---- Plots ----
    plot_loss_curves(
        {args.config_name: {'loss_curve': loss_curve}},
        output_dir=f'./plots/{args.config_name}',
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Config: {args.config_name}")
    print(f"Val  — AUC: {val_metrics['auc']:.4f}  F1: {val_metrics['f1']:.4f}  "
          f"Acc: {val_metrics['acc']:.4f}")
    if test_metrics:
        print(f"Test — AUC: {test_metrics['auc']:.4f}  F1: {test_metrics['f1']:.4f}  "
              f"Acc: {test_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
