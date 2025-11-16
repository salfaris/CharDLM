from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from nanodlm.dataset import CharacterLevelDataset


@dataclass
class TransformerConfig:
    """Base configuration for Transformer models."""

    smol: bool = True

    vocab_size: int | None = None
    block_size: int = None  # type: ignore

    n_embd: int = None  # type: ignore
    n_head: int = None  # type: ignore
    n_layer: int = None  # type: ignore
    dropout_rate: float = None  # type: ignore

    is_causal: bool = True

    def __post_init__(self):
        if self.block_size is None:
            self.block_size = 8 if self.smol else 256

        if self.n_embd is None:
            self.n_embd = 32 if self.smol else 384

        if self.n_head is None:
            self.n_head = 4 if self.smol else 6

        if self.n_layer is None:
            self.n_layer = 3 if self.smol else 6

        if self.dropout_rate is None:
            self.dropout_rate = 0.0 if self.smol else 0.2


@dataclass
class GPTConfig(TransformerConfig):
    """Configuration for GPT model."""

    pass


@dataclass
class DLMConfig(TransformerConfig):
    """Configuration for NanoDiffusionLM model."""

    smol: bool = True

    is_causal: bool = False
    diffusion_steps: int = 100
    mask_token_id: int | None = None

    # These will be set in __post_init__ based on smol value
    block_size: int = None  # type: ignore
    context_len: int = None  # type: ignore

    def __post_init__(self):
        # Set block_size based on smol if not explicitly provided
        if self.block_size is None:
            self.block_size = 128 if self.smol else 256

        # Set context_len based on smol if not explicitly provided
        if self.context_len is None:
            # Context length before which no tokens are masked. This is a hyperparam for the
            # expected prompt input. The prompt 'ROMEO: ' has context length 7. We are going
            # to freeze this to 16 for non-smol and 8 for smol models which is an arbitrary
            # choice.
            self.context_len = 8 if self.smol else 16

        # Validate context_len <= block_size
        assert (
            self.context_len <= self.block_size
        ), f"context_len ({self.context_len}) must be <= block_size ({self.block_size})"


class Buffer(nnx.Variable):
    pass


class Head(nnx.Module):
    """Single head of self-attention."""

    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.is_causal = config.is_causal

        head_size = config.n_embd // config.n_head

        self.key = nnx.Linear(config.n_embd, head_size, use_bias=False, rngs=rngs)
        self.query = nnx.Linear(config.n_embd, head_size, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(config.n_embd, head_size, use_bias=False, rngs=rngs)

        if self.is_causal:
            self.tril = Buffer(
                jnp.tril(jnp.ones((config.block_size, config.block_size)))
            )

        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cache=None):
        _, T, C = x.shape

        k = self.key(x)  # (B, T, C) = (B, T, head_size)
        q = self.query(x)  # (B, T, C) = (B, T, head_size)
        v = self.value(x)  # (B, T, C)

        # Compute attention scores ("affinities")
        # Alt use jnp.matrix_transpose(k) (designed to handle exactly this use case!)
        # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        # Scaled with C := head size so that softmax leads to diffused probas.
        wei = q @ k.transpose(0, -1, -2) * (C**-0.5)
        if self.is_causal:
            wei = jnp.where(self.tril[:T, :T] == 0, float("-inf"), wei)
        wei = nnx.softmax(wei, axis=-1)  # type: ignore
        wei = self.dropout(wei)  # Randomly prevent some nodes from communicating

        # Perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


class MultiHeadAttention(nnx.Module):
    """Mutli-head self-attention."""

    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.heads = nnx.Sequential(
            *[Head(config=config, rngs=rngs) for _ in range(config.n_head)]
        )
        self.proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        # dim = concat along axis -1 of num_heads Heads
        #     = (1, 1, num_heads) * (B, n_embd, head_size)
        #     = (1, 1, num_heads) * (B, n_embd, n_embd // num_heads)
        #     = (B, n_embd, n_embd)
        out = jnp.concat([h(x) for h in self.heads.layers], axis=-1)
        # Projection back into residual pathway
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nnx.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(config.n_embd, 4 * config.n_embd, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                4 * config.n_embd, config.n_embd, rngs=rngs
            ),  # Projection layer into residual pathway
            nnx.Dropout(config.dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jnp.ndarray):
        return self.net(x)


class Block(nnx.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.sa = MultiHeadAttention(config=config, rngs=rngs)
        self.ffwd = FeedForward(config=config, rngs=rngs)
        self.ln1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.block_size = config.block_size  # Use for generation.

        assert config.vocab_size is not None

        self.token_embedding_table = nnx.Embed(
            num_embeddings=config.vocab_size, features=config.n_embd, rngs=rngs
        )
        self.positional_embedding_table = nnx.Embed(
            num_embeddings=self.block_size, features=config.n_embd, rngs=rngs
        )
        self.blocks = nnx.Sequential(
            *[Block(config=config, rngs=rngs) for _ in range(config.n_layer)]
        )
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)  # Final layer norm
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, rngs=rngs)

    def __call__(self, idx: jnp.ndarray) -> jnp.ndarray:
        _, T = idx.shape

        # Think of logits as scores for the next char in the sequence.
        tok_emb = self.token_embedding_table(idx)  # (Batch, Time, Channel) = (B, T, C)
        pos_emb = self.positional_embedding_table(jnp.arange(T))  # (T, C)

        # (B, T, C) + (T, C) --> (B, T, C) + (1, T, C) = (B, T, C).
        # Note: not adding dimensions here, but we are showing how jax infers the batch
        # dimension in `pos_embd` and right-shift (T, C) -> (1, T, C) similar to in
        # pytorch.
        x = tok_emb + pos_emb
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

        return logits

    @nnx.jit
    def generate_step(self, padded_tokens, sample_index, rngs):
        logits = self(padded_tokens)

        # logits = logits[:, sample_index, :].squeeze(0)  # this is not jit-able
        logits = logits[0][sample_index]

        # sample_from equiv.
        next_token = rngs.categorical(logits)

        return next_token

    def generate_text(
        self,
        dataset: CharacterLevelDataset,
        max_tokens: int,
        start_tokens: list,
        rngs,
    ):
        generated = []

        print(dataset.decode(start_tokens), flush=True, end="")
        for _ in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1

            padded_tokens = (
                start_tokens
                + generated
                + [0] * (self.block_size - len(start_tokens) - len(generated))
            )
            padded_tokens = padded_tokens[-self.block_size :]  # Truncate to block size
            padded_tokens = jnp.array(padded_tokens).reshape(1, -1)

            next_token = int(self.generate_step(padded_tokens, sample_index, rngs))  # type: ignore
            generated.append(next_token)
            print(dataset.decode([next_token]), flush=True, end="")

        return dataset.decode(start_tokens + generated)

    @nnx.jit(static_argnums=(2,))
    def generate_fast(
        self, idx: jnp.ndarray, max_new_tokens: int, rngs: nnx.Rngs
    ) -> jnp.ndarray:
        block_size = self.block_size
        B, T = idx.shape

        # Pre-allocate output array with final size
        output_tokens = jnp.zeros((B, T + max_new_tokens), dtype=jnp.int32)
        output_tokens = output_tokens.at[:, :T].set(idx)

        # Extract the key ONCE before the scan
        initial_key = rngs()

        def generate_step(carry, step_idx):
            tokens, key = carry
            current_pos = T + step_idx

            key, subkey = jax.random.split(key)

            # Optionally pad on th e left
            effective_length = jnp.minimum(block_size, current_pos)
            start_idx = jnp.maximum(0, current_pos - block_size)
            # Crop idx to the last block_size tokens (since we are now doing pos encoding)
            # idx_cond = current_idx[:, -block_size:]
            idx_cond = jax.lax.dynamic_slice(tokens, (0, start_idx), (B, block_size))
            # If current_pos < block_size, idx_cond's right part should be padded on the left.

            # Pad idx_cond on the left if length < block_size
            idx_cond = jnp.where(
                jnp.arange(block_size) < block_size - effective_length, 0, idx_cond
            )

            # Get the predictions
            logits = self(idx_cond)  # dim = (B, C)

            # Focus only on the last time step (get idx -1 on Time index)
            logits = logits[:, -1, :]  # dim = (B, C)

            # jax.random.categorical is more similar to torch.multinomial.
            # Notice we don't require apply softmax to logits since rngs.categorical
            # expects logits rather than probabilities.
            # Also notice, reshape (B, 1) because cannot concat (B, T) with (B,), require
            # reshape to concat (B, T) with (B, 1) --> (B, T+1).
            # idx_next = jax.random.categorical(subkey, logits).reshape(
            #     logits.shape[0], 1
            # )  # dim = (B,) -> (B, 1)
            idx_next = jax.random.categorical(subkey, logits)

            # # Append sampled index to the running idx sequence
            # new_idx = jnp.concat([current_idx, idx_next], axis=1)  # dim = (B, T+1)
            tokens = tokens.at[:, current_pos].set(idx_next)

            return (tokens, key), None

        (final_tokens, _), _ = jax.lax.scan(
            generate_step,
            (
                output_tokens,
                initial_key,
            ),  # Note: idx is (B, T) array of indices in current context.
            jnp.arange(max_new_tokens),
        )

        return final_tokens


class NanoDiffusionLM(nnx.Module):
    def __init__(self, config: DLMConfig, rngs: nnx.Rngs):
        self.block_size = config.block_size  # Use for generation.
        self.context_len = config.context_len

        assert config.vocab_size is not None
        assert config.mask_token_id is not None
        assert self.context_len is not None

        self.mask_token_id = config.mask_token_id
        self.diffusion_steps = config.diffusion_steps
        # Linear schedule timestep
        self.mask_schedule = jnp.linspace(
            1.0 / self.diffusion_steps, 1.0, self.diffusion_steps
        )

        self.token_embedding_table = nnx.Embed(
            num_embeddings=config.vocab_size, features=config.n_embd, rngs=rngs
        )
        self.positional_embedding_table = nnx.Embed(
            num_embeddings=config.block_size, features=config.n_embd, rngs=rngs
        )
        self.time_embedding_table = nnx.Embed(
            num_embeddings=config.diffusion_steps, features=config.n_embd, rngs=rngs
        )
        self.blocks = nnx.Sequential(
            *[Block(config=config, rngs=rngs) for _ in range(config.n_layer)]
        )
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)  # Final layer norm
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, rngs=rngs)

    def __call__(self, idx: jnp.ndarray, time_idx: jnp.ndarray) -> jnp.ndarray:
        _, T = idx.shape

        # Think of logits as scores for the next char in the sequence.
        tok_emb = self.token_embedding_table(idx)  # (Batch, Time, Channel) = (B, T, C)
        pos_emb = self.positional_embedding_table(jnp.arange(T))  # (T, C)

        # Add time embeddings
        # time_idx.shape = (B,)
        time_emb = self.time_embedding_table(time_idx)  # (B, C)

        # (B, T, C) + (T, C) --> (B, T, C) + (1, T, C) = (B, T, C).
        # Note: not adding dimensions here, but we are showing how jax infers the batch
        # dimension in `pos_embd` and right-shift (T, C) -> (1, T, C) similar to in
        # pytorch.
        x = tok_emb + pos_emb

        # Broadcast time embedding over char dim T.
        # (B, T, C) + (B, C) -> (B, T, C) + (B, 1, C) -> (B, T, C)
        # Note: we are not adding dimensions here.
        x = x + time_emb[:, None, :]

        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

        return logits

    def corrupt_input(
        self, idx: jnp.ndarray, timesteps: jnp.ndarray, rngs: nnx.Rngs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Corrupt input tokens based on timesteps using masking strategy."""
        B, T = idx.shape

        prob_mask = self.mask_schedule[timesteps].reshape(-1, 1)  # Shape (B, 1)

        random_vals = rngs.uniform(shape=(B, T))
        mask = random_vals < prob_mask  # Shape (B, T), dtype=bool

        # Never mask the first block_size tokens
        mask = mask.at[:, : self.context_len].set(False)

        # Create corrupted input by replacing masked positions with mask token id
        corrupted_idx = jnp.where(mask, self.mask_token_id, idx)

        return corrupted_idx, mask

    def fast_dllm_decode(
        self,
        dataset: CharacterLevelDataset,
        prompt: list[int],  # aka context. prompt = context
        confidence_threshold: float,  # tau in the fast DLLM paper
        dllm_block_size: int | None = None,
        verbose: bool = True,
    ):
        # NOTE: To see the non-jittable original version, run:
        # git show ea37222:nanodlm/model.py | grep -A 100 "def fast_dllm_decode"

        # L in the fast DLLM paper
        new_tokens_len = self.block_size - len(prompt)
        assert new_tokens_len > 0

        if dllm_block_size is None:
            dllm_block_size = self.block_size // 2
        assert dllm_block_size <= self.block_size

        # This is K in the fast DLLM paper
        num_blocks = self.block_size // dllm_block_size

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

        if verbose:
            print(x)

        # Process each block sequentially
        # We can't use scan here easily due to dynamic slicing constraints
        # Instead, use a Python loop (or we could use fori_loop)
        for k in range(1, num_blocks + 1):
            s = prompt_len + (k - 1) * dllm_block_size
            e = min(prompt_len + k * dllm_block_size, self.block_size)

            # Use scan for the diffusion timesteps within each block
            def diffusion_step(x_inner, t):
                # Check if block is fully unmasked
                block_mask = x_inner[s:e] == dataset.mask_token_id
                still_has_masked = jnp.any(block_mask)

                def do_step(x_inner):
                    t_batch = jnp.clip(
                        jnp.full(shape=(1,), fill_value=t, dtype=jnp.int32),
                        0,
                        self.diffusion_steps - 1,
                    )

                    logits = self(x_inner.reshape(1, -1), t_batch).squeeze(0)

                    # Get masked positions within the block.
                    # Pad with size=dllm_block_size which handles final block where
                    # e-s < dllm_block_size. We use fill_value=-1 before adding s,
                    # so invalid positions become s-1 after the shift.
                    # Local because this is a mask within the x[s:e] block only.
                    masked_positions_local = jnp.nonzero(
                        block_mask, size=dllm_block_size, fill_value=-1
                    )[0]
                    num_masked = jnp.sum(block_mask)

                    # Add offset s AFTER checking validity to avoid s-1 ambiguity
                    # safe_positions uses local indices that are valid (>= 0)
                    safe_positions = jnp.where(
                        masked_positions_local >= 0, masked_positions_local + s, 0
                    )
                    # !!! Careful that we are using 0 as the dummy index for invalid
                    # positions since logits[safe_positions==0] means we will get logits
                    # for position 0 in the context which can lead to unpredictable
                    # predictions. Below we mask these values an non valid_mask and map
                    # these indices values to -jnp.inf. This implies that they will
                    # always fail the condfidence >= confidence_threshold check and so
                    # so will just be unmasked greedily.
                    logits_masked = logits[safe_positions]
                    confidence = jnp.max(nnx.softmax(logits_masked), axis=-1)

                    # Zero out confidence for padded positions
                    valid_mask = jnp.arange(dllm_block_size) < num_masked
                    confidence = jnp.where(valid_mask, confidence, -jnp.inf)

                    # Determine positions to unmask
                    high_conf = confidence >= confidence_threshold
                    has_high_conf = jnp.any(high_conf)

                    # If no high confidence, as in confidence==False for all positions,
                    # unmask the zeroth index in confidence (since jnp.argmax handles
                    # ties by returning the smallest index among ties).
                    max_conf_idx = jnp.argmax(confidence)
                    to_unmask = jnp.where(
                        has_high_conf,
                        high_conf,
                        jnp.arange(dllm_block_size) == max_conf_idx,
                    )

                    # Vectorized unmasking: get argmax for all positions at once
                    next_tokens = jnp.argmax(logits[safe_positions], axis=-1)

                    # Update x only at positions that should be unmasked and are valid
                    # Only update where to_unmask is True and position is valid
                    should_update = to_unmask & valid_mask
                    updates = jnp.where(
                        should_update, next_tokens, x_inner[safe_positions]
                    )
                    x_inner = x_inner.at[safe_positions].set(updates)

                    return x_inner

                # Use lax.cond instead of Python if
                x_inner = jax.lax.cond(still_has_masked, do_step, lambda x: x, x_inner)

                return x_inner, None

            # Scan over diffusion timesteps
            x, _ = jax.lax.scan(
                diffusion_step, x, jnp.arange(1, self.diffusion_steps + 1)
            )

            if verbose:
                print(x)

        return x
