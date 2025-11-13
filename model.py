import jax
import jax.numpy as jnp
import optax
from flax import nnx


class Buffer(nnx.Variable):
    pass


class Head(nnx.Module):
    """Single head of self-attention."""

    def __init__(
        self,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.key = nnx.Linear(n_embd, head_size, use_bias=False, rngs=rngs)
        self.query = nnx.Linear(n_embd, head_size, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(n_embd, head_size, use_bias=False, rngs=rngs)
        self.tril = Buffer(jnp.tril(jnp.ones((block_size, block_size))))

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, C) = (B, T, head_size)
        q = self.query(x)  # (B, T, C) = (B, T, head_size)

        # Compute attention scores ("affinities")
        # Alt use jnp.matrix_transpose(k) (designed to handle exactly this use case!)
        # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        # Scaled with C := head size so that softmax leads to diffused probas.
        wei = q @ k.transpose(0, -1, -2) * (C**-0.5)
        wei = jnp.where(self.tril[:T, :T] == 0, float("-inf"), wei)
        wei = nnx.softmax(wei, axis=-1)  # type: ignore
        wei = self.dropout(wei)  # Randomly prevent some nodes from communicating

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


class MultiHeadAttention(nnx.Module):
    """Mutli-head self-attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.heads = nnx.Sequential(
            *[
                Head(
                    head_size,
                    n_embd=n_embd,
                    block_size=block_size,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nnx.Linear(n_embd, n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

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

    def __init__(self, n_embd: int, dropout_rate: float, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(n_embd, 4 * n_embd, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                4 * n_embd, n_embd, rngs=rngs
            ),  # Projection layer into residual pathway
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jnp.ndarray):
        return self.net(x)


class Block(nnx.Module):
    """Transformer block: communication followed by computation."""

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        block_size: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.ffwd = FeedForward(n_embd, dropout_rate=dropout_rate, rngs=rngs)
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        num_heads: int,
        n_layer: int,
        block_size: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.block_size = block_size

        self.token_embedding_table = nnx.Embed(
            num_embeddings=vocab_size, features=n_embd, rngs=rngs
        )
        self.positional_embedding_table = nnx.Embed(
            num_embeddings=self.block_size, features=n_embd, rngs=rngs
        )
        self.sa_heads = MultiHeadAttention(
            num_heads=num_heads,
            head_size=n_embd // num_heads,
            n_embd=n_embd,
            block_size=block_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.blocks = nnx.Sequential(
            *[
                Block(
                    n_embd,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)  # Final layer norm
        self.lm_head = nnx.Linear(n_embd, vocab_size, rngs=rngs)

    def __call__(
        self, idx: jnp.ndarray, targets: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        B, T = idx.shape

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

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = jnp.mean(
                # Can verify softmax using the average ~= -log(1/VOCAB_SIZE) = 4.17.
                optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            )

        # !!! Logits is dim (B, T, C) if targets is None else (B*T, C)
        return logits, loss

    def generate(
        self, idx: jnp.ndarray, max_new_tokens: int, rngs: nnx.Rngs
    ) -> jnp.ndarray:
        # idx is (B, T) array of indices in current context.
        for _ in range(max_new_tokens):
            # Crop idx to the last self.block_size tokens (since we are now doing pos encoding)
            idx_cond = idx[:, -self.block_size :]

            # Get the predictions
            logits, _ = self(idx_cond)  # dim = (B, C)

            # Focus only on current_idx last time step (get idx -1 on Time index)
            logits = logits[:, -1, :]

            # jax.random.categorical is more similar to torch.multinomial.
            # Notice we don't require apply softmax to logits since rngs.categorical
            # expects logits rather than probabilities.
            # Also notice, reshape (B, 1) because cannot concat (B, T) with (B,), require
            # reshape to concat (B, T) with (B, 1) --> (B, T+1).
            idx_next = rngs.categorical(logits).reshape(
                logits.shape[0], 1
            )  # dim = (B,) -> (B, 1)

            # Append sampled index to the running idx sequence
            idx = jnp.concat([idx, idx_next], axis=1)  # dim = (B, T+1)

        return idx

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
            logits, _ = self(idx_cond)  # dim = (B, C)

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
