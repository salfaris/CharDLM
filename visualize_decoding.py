"""Visual simulation of the sequential decoding process in Fast-DLLM."""

import sys
import time
from typing import List

import jax.numpy as jnp
from flax import nnx

from chardlm.checkpoint import Checkpointer
from chardlm.dataset import load_shakespeare_dataset
from chardlm.model import CharDLM, DLMConfig


def format_sequence(
    tokens: List[int],
    dataset,
    prompt_len: int = 0,
    block_ranges: List[tuple] | None = None,
    block_colors: List[str] | None = None,
) -> str:
    """Format sequence for display with color coding.

    - Cyan for prompt tokens
    - Red for masked tokens
    - Block-specific colors for decoded tokens based on which block decoded them
    - Blue for newlines
    """
    result = []

    for i, token in enumerate(tokens):
        if token == dataset.mask_token_id:
            result.append("\033[91m[#]\033[0m")  # Red for masks
        else:
            char = dataset.decode([token])

            # Determine color based on position
            if i < prompt_len:
                color = "\033[96m"  # Cyan for prompt
            elif block_ranges and block_colors:
                # Find which block this position belongs to
                color = "\033[92m"  # Default green
                for block_idx, (start, end) in enumerate(block_ranges):
                    if start <= i < end:
                        color = block_colors[block_idx % len(block_colors)]
                        break
            else:
                color = "\033[92m"  # Default green for decoded

            if char == "\n":
                result.append(f"{color}[↵]\033[0m")  # Colored newlines
            elif char == " ":
                result.append(f"{color} \033[0m")  # Colored spaces
            else:
                result.append(f"{color}{char}\033[0m")  # Colored characters
    return "".join(result)


def print_sequence_inplace(
    tokens: List[int],
    dataset,
    prompt_len: int = 0,
    is_first: bool = False,
    block_header: str | None = None,
    block_ranges: List[tuple] | None = None,
    block_colors: List[str] | None = None,
):
    """Print sequence in-place, clearing screen before each update."""
    if not is_first:
        # Clear screen and move cursor to top
        sys.stdout.write("\033[2J\033[H")

    # Print block header if provided
    if block_header:
        print(block_header)

    sequence_str = format_sequence(
        tokens, dataset, prompt_len, block_ranges, block_colors
    )
    print(sequence_str)
    sys.stdout.flush()


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'═'*80}")
    print(f"  {text}")
    print(f"{'═'*80}")


def print_block_header(block_num: int, start: int, end: int):
    """Print block information."""
    print(f"\n┌{'─'*78}┐")
    print(
        f"│ 🔄 BLOCK {block_num} (positions {start}-{end-1}){' '*(50-len(str(block_num))-len(str(start))-len(str(end)))}│"
    )
    print(f"└{'─'*78}┘")


def visualize_decoding(
    model: CharDLM,
    dataset,
    prompt: List[int],
    dllm_block_size: int,
    confidence_threshold: float = 0.9,
    delay: float = 0.5,
):
    """Visualize the sequential block-wise decoding process."""

    # Initialize sequence
    new_tokens_len = model.block_size - len(prompt)
    prompt_tokens = jnp.array(prompt)
    prompt_len = len(prompt_tokens)

    x = jnp.concatenate(
        [
            prompt_tokens,
            jnp.full(
                shape=(new_tokens_len,),
                fill_value=dataset.mask_token_id,
                dtype=jnp.int32,
            ),
        ]
    )

    num_blocks = model.block_size // dllm_block_size

    print(f"\n📝 Initial State:")
    print(f"  Prompt: \033[96m{dataset.decode(prompt)}\033[0m\n")
    print_sequence_inplace(x.tolist(), dataset, prompt_len, is_first=True)
    time.sleep(delay * 1.0)

    # Track which blocks have been decoded for coloring
    block_ranges = []  # List of (start, end) tuples for each completed block

    # Process each block sequentially
    # Colors for different blocks (cycling through)
    block_colors = [
        "\033[95m",  # Magenta
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[91m",  # Red
        "\033[96m",  # Cyan
    ]

    for k in range(1, num_blocks + 1):
        s = prompt_len + (k - 1) * dllm_block_size
        e = min(prompt_len + k * dllm_block_size, model.block_size)

        # Add current block range for coloring
        current_block_ranges = block_ranges + [(s, e)]

        # Create block header string with color
        block_color = block_colors[(k - 1) % len(block_colors)]
        block_header = f"\n{block_color}┌{'─'*78}┐\n│ 🔄 BLOCK {k} (positions {s}-{e-1}){' '*(54-len(str(k))-len(str(s))-len(str(e)))}│\n└{'─'*78}┘\033[0m\n"
        time.sleep(delay * 0.1)

        for t in range(model.diffusion_steps):
            # Check if block is fully unmasked
            block_mask = x[s:e] == dataset.mask_token_id
            still_has_masked = jnp.any(block_mask)

            if not still_has_masked:
                break

            # Run model at current timestep
            t_batch = jnp.clip(
                jnp.full(shape=(1,), fill_value=t, dtype=jnp.int32),
                0,
                model.diffusion_steps - 1,
            )
            logits = model(x.reshape(1, -1), t_batch).squeeze(0)

            # Get masked positions within the block
            masked_positions_local = jnp.nonzero(
                block_mask, size=dllm_block_size, fill_value=-1
            )[0]
            num_masked = jnp.sum(block_mask)

            # Add offset s to get global positions
            safe_positions = jnp.where(
                masked_positions_local >= 0, masked_positions_local + s, 0
            )

            # Get logits for masked positions
            logits_masked = logits[safe_positions]
            confidence = jnp.max(nnx.softmax(logits_masked), axis=-1)

            # Zero out confidence for padded positions
            valid_mask = jnp.arange(dllm_block_size) < num_masked
            confidence = jnp.where(valid_mask, confidence, -jnp.inf)

            # Determine positions to unmask based on confidence threshold
            high_conf = confidence >= confidence_threshold
            has_high_conf = jnp.any(high_conf)

            # If no high confidence, unmask the position with highest confidence
            max_conf_idx = jnp.argmax(confidence)
            to_unmask = jnp.where(
                has_high_conf, high_conf, jnp.arange(dllm_block_size) == max_conf_idx
            )

            # Vectorized unmasking: get argmax for all positions at once
            next_tokens = jnp.argmax(logits[safe_positions], axis=-1)

            # Store old state before update
            x_before = x.copy()

            # Update x only at positions that should be unmasked and are valid
            should_update = to_unmask & valid_mask
            updates = jnp.where(should_update, next_tokens, x[safe_positions])
            x = x.at[safe_positions].set(updates)

            # Count how many tokens were just unmasked
            num_just_unmasked = int(jnp.sum(should_update))

            # Update display whenever tokens are unmasked
            if num_just_unmasked > 0:
                time.sleep(delay * 0.08)
                print_sequence_inplace(
                    x.tolist(),
                    dataset,
                    prompt_len,
                    is_first=False,
                    block_header=block_header,
                    block_ranges=current_block_ranges,
                    block_colors=block_colors,
                )

        # Add completed block range to tracking
        block_ranges.append((s, e))

        # Add spacing after block completes
        print()
        time.sleep(delay * 0.15)

        # Check if entire sequence is fully decoded (early termination)
        remaining_mask = x[prompt_len:] == dataset.mask_token_id
        if not jnp.any(remaining_mask):
            blocks_skipped = num_blocks - k
            if blocks_skipped > 0:
                print(
                    f"\n⚡ Early termination: All tokens decoded! Skipping {blocks_skipped} remaining block(s)."
                )
            break

    print_header("✨ DECODING COMPLETE!")

    print(f"\n📝 Original Prompt:")
    print(f"{'─'*80}")
    print(dataset.decode(x[:prompt_len].tolist()))
    print(f"{'─'*80}")

    print(f"\n📄 Generated Text:")
    print(f"{'─'*80}")
    print(dataset.decode(x[prompt_len:].tolist()))
    print(f"{'─'*80}")

    return x


def main():
    """Run the visualization."""

    # Setup
    print("\n🚀 Initializing model and dataset...")

    rngs = nnx.Rngs(44)
    dataset = load_shakespeare_dataset()

    smol = False
    # ckpt_name = "chardlm-smol" if smol else "chardlm-big"
    ckpt_name = "chardlm-big-256block-randomUnmaskedContextLen"
    checkpointer = Checkpointer(name=ckpt_name)

    config = DLMConfig(
        smol=smol,
        vocab_size=dataset.vocab_size,
        mask_token_id=dataset.mask_token_id,
    )

    model = CharDLM(config, rngs=rngs)
    model = checkpointer.load_model_only(model)
    model.eval()

    print("✓ Model loaded successfully!")

    # Get a random sample from training set
    x_batch, _ = dataset.get_batch_jit(
        rngs=rngs,
        split="train",
        num_samples=1,
        batch_size=10,
        block_size=config.block_size,
    )
    full_sequence = x_batch.squeeze(0)  # Remove batch dimensions
    full_sequence = full_sequence[7]  # Take first sample in batch
    prompt_len = 100  # Use first 100 tokens as prompt
    prompt = full_sequence[:prompt_len].tolist()

    # Use smaller block size for better visualization
    dllm_block_size = model.block_size // 4

    visualize_decoding(
        model=model,
        dataset=dataset,
        prompt=prompt,
        dllm_block_size=dllm_block_size,
        confidence_threshold=0.3,
        delay=0.5,  # Delay between steps in seconds
    )


if __name__ == "__main__":
    main()
