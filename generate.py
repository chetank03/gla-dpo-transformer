import argparse
import torch
import tiktoken
import os
import sys

try:
    from gla_model import GLATransformer, generate_text
except ImportError:
    print("Error: Could not import GLATransformer or generate_text.")
    print("Please ensure gla_model.py is in the current directory.")
    sys.exit(1)


def infer_config(state_dict, checkpoint, args):
    """Recovers the model architecture from a checkpoint.

    vocab_size, d_model, block_size and n_blocks are read from tensor shapes, which is
    authoritative. n_heads and use_pre_norm cannot be recovered from shapes (q_proj is
    d_model x d_model for any head count dividing d_model), so they come from the
    'args' dict the pretraining script saves, and fall back to the CLI value.
    """
    saved = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if not isinstance(saved, dict):
        saved = {}

    vocab_size, d_model = state_dict["token_embedding.weight"].shape
    block_size = state_dict["positional_embedding.weight"].shape[0]
    n_blocks = 1 + max(
        int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")
    )

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "block_size": int(block_size),
        "n_blocks": int(n_blocks),
        "n_heads": int(saved.get("n_heads", args.n_heads)),
        "use_pre_norm": bool(saved.get("use_pre_norm", args.use_pre_norm)),
    }


def config_source(state_dict, checkpoint):
    """Describes where n_heads and use_pre_norm came from, for the startup log."""
    saved = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if isinstance(saved, dict) and "n_heads" in saved:
        return "shapes + checkpoint args"
    return "shapes + CLI defaults for n_heads and use_pre_norm"


def parse_args():
    """Parses command-line arguments for model loading and generation."""
    parser = argparse.ArgumentParser(
        description="Run inference on a DPO-trained Transformer model."
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the saved DPO-trained model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a",
        help="Starting phrase for text generation. Default='Once upon a'."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate. Default=50."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling probability. Set to None for greedy decoding. Default=0.95."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier (e.g., 'cuda:0', 'cpu')."
    )

    # Model configuration
    parser.add_argument(
        "--embed_size",
        type=int,
        default=1024,
        help="Dimension of the model (d_model). Default=1024."
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=4,
        help="Number of Transformer blocks. Default=4."
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads. Default=8."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Maximum sequence length. Default=1024."
    )
    parser.add_argument(
        "--use_pre_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pre-normalization (must match training config). Default=True."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    print(f"Device: {device}")
    print(f"Vocab Size: {vocab_size}")

    # Load checkpoint
    if not os.path.exists(args.load_path):
        print(f"Error: Checkpoint file not found at {args.load_path}")
        return

    try:
        checkpoint = torch.load(args.load_path, map_location=device)
        
        if "policy_model_state_dict" in checkpoint:
            state_dict = checkpoint["policy_model_state_dict"]
            print(f"Loaded DPO policy model state dict from {args.load_path}")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"Loaded model state dict from {args.load_path}")
        else:
            state_dict = checkpoint
            print(f"Loaded state dict from {args.load_path}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Instantiate and load model.
    # The architecture is read off the checkpoint rather than taken from the CLI
    # defaults, because a checkpoint whose shapes disagree with the defaults fails to
    # load. Anything the checkpoint cannot tell us falls back to the CLI value.
    config = infer_config(state_dict, checkpoint, args)
    print(
        "Model config: "
        + ", ".join(f"{k}={v}" for k, v in config.items())
        + f" (source: {config_source(state_dict, checkpoint)})"
    )

    try:
        model = GLATransformer(**config).to(device)

        model.load_state_dict(state_dict)
        model.eval()

        print(f"Model loaded successfully.")
        print("-" * 50)

    except RuntimeError as e:
        print(f"\nError loading state_dict: {e}")
        print("Please verify that --embed_size, --n_blocks, --n_heads, --block_size, and --use_pre_norm match the training configuration.")
        return

    # Generate text
    print(f"Prompt: '{args.prompt}'")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Sampling: {'Top-P (p=' + str(args.top_p) + ')' if args.top_p is not None else 'Greedy'}")
    print("-" * 50)

    try:
        generated_text = generate_text(
            model=model,
            enc=enc,
            init_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            top_p=args.top_p,
        )

        print("\n--- GENERATED TEXT ---")
        print(generated_text)
        print("----------------------\n")

    except Exception as e:
        print(f"\nAn error occurred during text generation: {e}")
        return


if __name__ == "__main__":
    main()
