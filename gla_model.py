"""
Gated Linear Attention (GLA) Transformer Model
Implementation based on "Gated Linear Attention Transformers with Hardware-Efficient Training"

This module implements a complete transformer architecture using Gated Linear Attention,
including model definition, training, and text generation capabilities.
"""
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import load_dataset
import tiktoken

# Command-line arg parsing

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GLA Transformer on TinyStories and/or custom text files."
    )
    parser.add_argument(
        "--input_files",
        nargs="*",
        default=None,
        help="Optional list of text files to mix in as data sources.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="roneneldan/TinyStories",
        help="HuggingFace dataset path to use as main dataset (default: roneneldan/TinyStories)",
    )
    parser.add_argument(
        "--hf_weight",
        type=float,
        default=0.5,
        help="Probability of sampling from HuggingFace dataset. Default=0.5.",
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=None,
        help="If set, each epoch ends after this many steps.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Maximum sequence length. Default=512.",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=1024,
        help="Embedding dimension. Default=512.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt for generation. Default='Once upon a time'.",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        default="cuda:0",
        help="Device identifier. Default='cuda:0'.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="gla_model.pth",
        help="Path to save model.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load model checkpoint.",
    )
    parser.add_argument(
        "--use_pre_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pre-normalization (recommended for GLA).",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads. Default=8.",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=6,
        help="Number of transformer blocks. Default=6.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size. Default=16.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Training epochs. Default=3.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate. Default=3e-4.",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value. Default=1.0.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Learning rate warmup steps. Default=100.",
    )

    args = parser.parse_args()
    return args


# Data handling

class MixedSequenceDataset(torch.utils.data.Dataset):
    """Dataset that mixes sequences from HuggingFace dataset and custom files."""
    
    def __init__(self, hf_sequences, custom_sequences, hf_sampling_prob: float):
        super().__init__()
        self.hf_sequences = hf_sequences
        self.custom_sequences = custom_sequences
        self.hf_sampling_prob = hf_sampling_prob

        self.has_hf_data = len(self.hf_sequences) > 0
        self.has_custom_data = len(self.custom_sequences) > 0
        self.total_length = len(self.hf_sequences) + len(self.custom_sequences)

        if self.total_length == 0:
            raise ValueError("No data found!")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        random_value = random.random()
        if self.has_hf_data and self.has_custom_data:
            if random_value < self.hf_sampling_prob:
                seq_idx = random.randint(0, len(self.hf_sequences) - 1)
                sequence = self.hf_sequences[seq_idx]
            else:
                seq_idx = random.randint(0, len(self.custom_sequences) - 1)
                sequence = self.custom_sequences[seq_idx]
        elif self.has_hf_data:
            seq_idx = random.randint(0, len(self.hf_sequences) - 1)
            sequence = self.hf_sequences[seq_idx]
        else:
            seq_idx = random.randint(0, len(self.custom_sequences) - 1)
            sequence = self.custom_sequences[seq_idx]

        return torch.tensor(sequence, dtype=torch.long)


def sequence_collate_fn(batch):
    """Collate function that pads sequences to the same length."""
    max_length = max(len(sequence) for sequence in batch)
    batch_size = len(batch)
    padded_batch = torch.zeros(max_length, batch_size, dtype=torch.long)

    for batch_idx, sequence in enumerate(batch):
        sequence_length = sequence.size(0)
        padded_batch[:sequence_length, batch_idx] = sequence

    return padded_batch


# Loss computation

def compute_next_token_loss(logits, target_tokens):
    """Compute cross-entropy loss for next token prediction."""
    sequence_length, batch_size, vocab_size = logits.shape
    if sequence_length < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    predictions = logits[:-1, :, :].reshape(-1, vocab_size)
    targets = target_tokens[1:, :].reshape(-1)
    return F.cross_entropy(predictions, targets, ignore_index=0)


# RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return (x / (rms + self.eps)) * self.scale


# Gated Linear Attention Implementation

class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention with data-dependent gates.

    Uses the formulation: S_t = (α_t^T ⊗ 1) ⊙ S_{t-1} + k_t^T v_t
    where α_t are learned scalar gates per dimension.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Key dimension per head (same as value)
        self.d_v = d_model // n_heads  # Value dimension per head

        # QKV projections
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_v, bias=False)

        # Gate projection (low-rank for efficiency)
        self.gate_proj1 = nn.Linear(d_model, 16, bias=True)
        self.gate_proj2 = nn.Linear(16, n_heads * self.d_k, bias=True)
        self.gate_temp = 16.0  # Temperature for slower forgetting

        # Output gating and projection
        self.output_gate_proj = nn.Linear(d_model, n_heads * self.d_v, bias=True)
        self.out_proj = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        self.eps = 1e-6

    def forward(self, x, cache=None):
        """
        Args:
            x: (seq_len, batch, d_model)
            cache: hidden state S_t (B, n_heads, d_k, d_v) or None

        Returns:
            out: (seq_len, batch, d_model)
            new_cache: updated hidden state
        """
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        B, T, C = x.size()

        # Compute Q, K, V
        q = (
            self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        )  # (B, H, T, d_k)
        k = (
            self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        )  # (B, H, T, d_k)
        v = (
            self.v_proj(x).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        )  # (B, H, T, d_v)

        # Compute gates α_t (per-dimension forget gates)
        gate_hidden = F.silu(self.gate_proj1(x))  # (B, T, 16)
        alpha = torch.sigmoid(
            self.gate_proj2(gate_hidden) / self.gate_temp
        )  # (B, T, H*d_k)
        alpha = alpha.view(B, T, self.n_heads, self.d_k).transpose(
            1, 2
        )  # (B, H, T, d_k)

        # Initialize or retrieve hidden state
        if cache is not None:
            S = cache  # (B, H, d_k, d_v)
        else:
            S = torch.zeros(
                B, self.n_heads, self.d_k, self.d_v, device=x.device, dtype=x.dtype
            )

        outputs = []

        # Process each timestep with recurrent update
        for t in range(T):
            q_t = q[:, :, t : t + 1, :]  # (B, H, 1, d_k)
            k_t = k[:, :, t : t + 1, :]  # (B, H, 1, d_k)
            v_t = v[:, :, t : t + 1, :]  # (B, H, 1, d_v)
            alpha_t = alpha[:, :, t, :]  # (B, H, d_k)

            # Gated recurrent update: S_t = diag(α_t) @ S_{t-1} + k_t^T @ v_t
            # This is the core of GLA - fixed-size state that gets updated
            S = alpha_t.unsqueeze(-1) * S + k_t.transpose(-2, -1) @ v_t

            # Compute output: o_t = q_t @ S_t
            o_t = q_t @ S  # (B, H, 1, d_v)
            outputs.append(o_t)

        # Stack outputs
        out = torch.cat(outputs, dim=2)  # (B, H, T, d_v)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_v)

        # Apply LayerNorm to each head's output (per-head normalization from paper)
        out_normalized = []
        for h in range(self.n_heads):
            start_idx = h * self.d_v
            end_idx = (h + 1) * self.d_v
            head_out = out[:, :, start_idx:end_idx]
            head_out = F.layer_norm(head_out, (self.d_v,))
            out_normalized.append(head_out)
        out = torch.cat(out_normalized, dim=-1)

        # Output gating (SwiGLU-style gating for attention output)
        output_gate = F.silu(self.output_gate_proj(x))
        out = out * output_gate

        # Final projection
        out = self.out_proj(out)

        # Return updated hidden state as cache
        new_cache = S

        return out.transpose(0, 1), new_cache  # (T, B, C), cache


# Transformer Block with GLA

class GLATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, expansion_factor=4, use_pre_norm=True):
        super().__init__()

        self.use_pre_norm = use_pre_norm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.attn = GatedLinearAttention(d_model, n_heads)

        # SwiGLU FFN
        d_hidden = d_model * expansion_factor
        self.ffn_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.ffn_up = nn.Linear(d_model, d_hidden, bias=False)
        self.ffn_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x, cache=None):
        if self.use_pre_norm:
            # Pre-norm: norm before attention/FFN (recommended for GLA)
            attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
            x = x + attn_out

            # FFN with SwiGLU
            normed = self.norm2(x)
            ffn_out = F.silu(self.ffn_gate(normed)) * self.ffn_up(normed)
            ffn_out = self.ffn_down(ffn_out)
            x = x + ffn_out
        else:
            # Post-norm: norm after residual connection
            attn_out, new_cache = self.attn(x, cache=cache)
            x = self.norm1(x + attn_out)

            # FFN with SwiGLU
            ffn_out = F.silu(self.ffn_gate(x)) * self.ffn_up(x)
            ffn_out = self.ffn_down(ffn_out)
            x = self.norm2(x + ffn_out)

        return x, new_cache


# Complete GLA Transformer Model

class GLATransformer(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_heads=8,
        n_blocks=6,
        block_size=512,
        use_pre_norm=True,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(block_size, d_model)
        self.block_size = block_size
        self.use_pre_norm = use_pre_norm

        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [
                GLATransformerBlock(d_model, n_heads, use_pre_norm=use_pre_norm)
                for _ in range(n_blocks)
            ]
        )

        self.norm_final = RMSNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.unembedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens_seq, cache=None):
        T, B = tokens_seq.shape
        device = tokens_seq.device

        token_emb = self.token_embedding(tokens_seq)

        pos_offset = 0
        position_ids = (
            torch.arange(pos_offset, pos_offset + T, device=device)
            .unsqueeze(1)
            .expand(T, B)
        )
        pos_emb = self.positional_embedding(position_ids)

        x = self.dropout(token_emb + pos_emb)

        new_cache = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            x, block_new_cache = block(x, cache=block_cache)
            new_cache.append(block_new_cache)

        x = self.norm_final(x)
        logits = self.unembedding(x)

        return logits, new_cache


# Improved Text Generation

def nucleus_sampling(logits, p=0.95, temperature=1.0):
    """Enhanced nucleus sampling with temperature control."""
    if temperature != 1.0:
        logits = logits / temperature

    if p >= 1.0:
        probs = F.softmax(logits, dim=-1)
        sampled_token = torch.multinomial(probs, num_samples=1)
        return sampled_token.item()

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_mask = cumulative_probs >= p
    if cutoff_mask.any():
        cutoff_index = cutoff_mask.nonzero(as_tuple=True)[0][0].item() + 1
    else:
        cutoff_index = len(sorted_probs)

    top_probs = sorted_probs[:cutoff_index]
    top_indices = sorted_indices[:cutoff_index]
    top_probs = top_probs / top_probs.sum()

    sampled_idx = torch.multinomial(top_probs, num_samples=1)
    sampled_token = top_indices[sampled_idx]

    return sampled_token.item()


def generate_text(
    model, enc, init_text, max_new_tokens=50, device="cpu", top_p=0.95, temperature=0.8
):
    """Generate text using the model with KV caching for efficiency."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        kv_cache = None

        # Process initial context
        sequence_tensor = torch.tensor(
            context_tokens, dtype=torch.long, device=device
        ).unsqueeze(1)
        logits_sequence, kv_cache = model(sequence_tensor, cache=kv_cache)
        next_token_logits = logits_sequence[-1, 0, :]

        # Generate new tokens
        for step in range(max_new_tokens):
            if top_p is None:
                selected_token = torch.argmax(next_token_logits).item()
            else:
                selected_token = nucleus_sampling(
                    next_token_logits, p=top_p, temperature=temperature
                )

            context_tokens.append(selected_token)

            if step < max_new_tokens - 1:
                new_token_tensor = torch.tensor(
                    [selected_token], dtype=torch.long, device=device
                ).unsqueeze(1)
                logits_sequence, kv_cache = model(new_token_tensor, cache=kv_cache)
                next_token_logits = logits_sequence[-1, 0, :]

    model.train(was_training)
    generated_text = enc.decode(context_tokens)
    return generated_text


# Evaluation

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_tokens in test_loader:
            batch_tokens = batch_tokens.to(device)
            logits, _ = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            total_loss += loss.item()
            total_batches += 1

    avg_test_loss = total_loss / total_batches if total_batches > 0 else 0.0
    model.train()
    return avg_test_loss


# Training

def compute_learning_rate(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    """Compute learning rate using cosine annealing schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    device,
    learning_rate=3e-4,
    log_interval=100,
    sample_interval=50,
    max_steps_per_epoch=None,
    tokenizer=None,
    sample_prompt="Once upon a time",
    gradient_clip=1.0,
    warmup_steps=100,
):
    """Train model with gradient clipping, learning rate warmup, and cosine annealing."""
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    global_step = 0
    max_steps = num_epochs * (max_steps_per_epoch or len(train_loader))

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        window_loss = 0.0
        window_count = 0
        steps_in_epoch = 0

        for batch_idx, token_batch in enumerate(train_loader, start=1):
            steps_in_epoch += 1
            global_step += 1

            # Learning rate schedule
            current_lr = compute_learning_rate(global_step, warmup_steps, max_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            token_batch = token_batch.to(device)
            logits, _ = model(token_batch)
            loss = compute_next_token_loss(logits, token_batch)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            epoch_total_loss += loss.item()
            window_loss += loss.item()
            window_count += 1

            if batch_idx % log_interval == 0:
                avg_window_loss = window_loss / window_count
                print(
                    f"Epoch {epoch}/{num_epochs}, Step {batch_idx}, Loss: {avg_window_loss:.4f}, LR: {current_lr:.6f}"
                )
                window_loss = 0.0
                window_count = 0

            # Generation samples
            if (batch_idx == 1 or batch_idx % sample_interval == 0) and tokenizer is not None:
                with torch.no_grad():
                    sample_text = generate_text(
                        model,
                        enc=tokenizer,
                        init_text=sample_prompt,
                        max_new_tokens=50,
                        device=device,
                        top_p=0.9,
                        temperature=0.8,
                    )
                    print(f"\nSample: {sample_text}\n")

            if max_steps_per_epoch and steps_in_epoch >= max_steps_per_epoch:
                break

        avg_epoch_loss = epoch_total_loss / steps_in_epoch
        test_loss = evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch} - Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

    return global_step


# Main

def main():
    args = parse_args()

    device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: Gated Linear Attention (GLA) Transformer")

    # Data loading
    hf_sequences = []
    custom_sequences = []

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    if args.hf_weight > 0.0:
        print(f"Loading HuggingFace dataset: {args.hf_dataset} ...")
        dataset = load_dataset(args.hf_dataset, split="train")
        dataset = dataset.select(range(min(20000, len(dataset))))

        for sample in dataset:
            text_data = sample.get("text")

            if isinstance(text_data, str) and text_data:
                tokens = tokenizer.encode(text_data)[: args.block_size]
                if len(tokens) > 0:
                    hf_sequences.append(tokens)
            else:
                print(f"Skipping non-string/empty sample: {text_data}")

    if args.input_files:
        for filepath in args.input_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = tokenizer.encode(line.strip())[: args.block_size]
                    if len(tokens) > 0:
                        custom_sequences.append(tokens)

    combined_dataset = MixedSequenceDataset(
        hf_sequences, custom_sequences, args.hf_weight
    )

    train_size = int(0.9 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=sequence_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sequence_collate_fn,
    )

    # Model creation
    model = GLATransformer(
        vocab_size=vocab_size,
        d_model=args.embed_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        block_size=args.block_size,
        use_pre_norm=args.use_pre_norm,
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  Embedding dim: {args.embed_size}")
    print(f"  Heads: {args.n_heads}")
    print(f"  Blocks: {args.n_blocks}")
    print(f"  Pre-normalization: {args.use_pre_norm}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load or train
    if args.load_path:
        print(f"Loading from {args.load_path}...")
        checkpoint = torch.load(args.load_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        print(f"Generating with prompt: '{args.prompt}'")
        for temp in [0.7, 0.9, 1.0]:
            text = generate_text(
                model,
                enc=tokenizer,
                init_text=args.prompt,
                max_new_tokens=100,
                device=device,
                top_p=0.95,
                temperature=temp,
            )
            print(f"\nTemperature {temp}:\n{text}\n{'-'*60}")
    else:
        # Training
        print("Starting training...\n")
        train_model(
            model,
            train_loader,
            test_loader,
            args.num_epochs,
            device,
            learning_rate=args.learning_rate,
            log_interval=100,
            sample_interval=50,
            max_steps_per_epoch=args.max_steps_per_epoch,
            tokenizer=tokenizer,
            sample_prompt=args.prompt,
            gradient_clip=args.gradient_clip,
            warmup_steps=args.warmup_steps,
        )

        # Save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            },
            args.save_path,
        )
        print(f"\nModel saved to {args.save_path}")

        # Final generation
        print(f"\nFinal generation:")
        for temp in [0.7, 0.9]:
            text = generate_text(
                model,
                enc=tokenizer,
                init_text=args.prompt,
                max_new_tokens=100,
                device=device,
                top_p=0.95,
                temperature=temp,
            )
            print(f"Temp {temp}: {text}\n")

    print("\n*** Training complete! ***")


if __name__ == "__main__":
    main()
