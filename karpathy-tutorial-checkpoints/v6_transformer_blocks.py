import time
from typing import Literal

import flax
import jax
import jax.numpy as jnp
import optax
from flax import nnx

print(f"flax: {flax.__version__}")
print(f"jax: {jax.__version__}")
print(f"optax: {optax.__version__}")


# Hyperparameters
BATCH_SIZE = 32  # How many independent sequences will be process in parallel?
BLOCK_SIZE = 8  # What is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
EVAL_ITERS = 200
N_EMBD = 32  # Create a level of interaction

rngs = nnx.Rngs(1337)

# !curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get all the unique characters in the text.
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
# Map chars to ints.
s2i = {ch: i for i, ch in enumerate(chars)}
i2s = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: "".join([i2s[i] for i in l])

# Train test splits
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(
    rngs: nnx.Rngs,
    split: Literal["train", "val"],
    block_size: int,
    batch_size: int,
):
    """Shakespeare data loader."""

    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data

    maxval = len(data) - block_size
    start_indices = rngs.randint(shape=(batch_size,), minval=0, maxval=maxval)

    x = jnp.stack([data[i : i + block_size] for i in start_indices])
    y = jnp.stack([data[i + 1 : i + 1 + block_size] for i in start_indices])

    return x, y


class Buffer(nnx.Variable):
    pass


class Head(nnx.Module):
    """Single head of self-attention."""

    def __init__(self, head_size: int, rngs: nnx.Rngs):
        self.key = nnx.Linear(N_EMBD, head_size, use_bias=False, rngs=rngs)
        self.query = nnx.Linear(N_EMBD, head_size, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(N_EMBD, head_size, use_bias=False, rngs=rngs)
        self.tril = Buffer(jnp.tril(jnp.ones((BLOCK_SIZE, BLOCK_SIZE))))

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
        wei = nnx.softmax(wei, axis=-1)

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out


class MultiHeadAttention(nnx.Module):
    """Mutli-head self-attention."""

    def __init__(self, num_heads: int, head_size: int, rngs: nnx.Rngs):
        self.heads = nnx.Sequential(
            *[Head(head_size, rngs=rngs) for _ in range(num_heads)]
        )

    def __call__(self, x: jnp.ndarray):
        return jnp.concat([h(x) for h in self.heads.layers], axis=-1)


class FeedForward(nnx.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd: int, rngs: nnx.Rngs):
        self.net = nnx.Sequential(nnx.Linear(n_embd, n_embd, rngs=rngs), nnx.relu)

    def __call__(self, x: jnp.ndarray):
        return self.net(x)


class Block(nnx.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, rngs: nnx.Rngs):
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, rngs=rngs)
        self.ffwd = FeedForward(n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = self.sa(x)
        x = self.ffwd(x)
        return x


class BigramLanguageModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.token_embedding_table = nnx.Embed(
            num_embeddings=VOCAB_SIZE, features=N_EMBD, rngs=rngs
        )
        self.positional_embedding_table = nnx.Embed(
            num_embeddings=BLOCK_SIZE, features=N_EMBD, rngs=rngs
        )
        self.sa_heads = MultiHeadAttention(4, N_EMBD // 4, rngs=rngs)
        self.blocks = nnx.Sequential(
            Block(N_EMBD, n_head=4, rngs=rngs),
            Block(N_EMBD, n_head=4, rngs=rngs),
            Block(N_EMBD, n_head=4, rngs=rngs),
        )
        self.lm_head = nnx.Linear(N_EMBD, VOCAB_SIZE, rngs=rngs)

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
            # Crop idx to the last BLOCK_SIZE tokens (since we are now doing pos encoding)
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Get the predictions
            logits, _ = self(idx_cond)  # dim = (B, C)

            # Focus only on the last time step (get idx -1 on Time index)
            logits = logits[:, -1, :]  # dim = (B, C)

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


# We don't jit `loss_fn` because we will be using it within the training (train_step)
# and eval step (estimate_loss). When jit compiles the training/eval step, it traces
# through all operations inside the function, which in this case includes `loss_fn`.
# This reduces trace complexity.
def loss_fn(model, idx, targets):
    logits, loss = model(idx, targets)
    return loss, logits


# We are using JAX, we don't require a context manager `torch.no_grad()` since
# JAX do not track gradients unless explicitly asked via nnx.grad or equiv.
def estimate_loss(rngs: nnx.Rngs, model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = jnp.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            # Because of dynamic index slicing in `get_batch`, we can't jit the
            # `estimate_loss` function.
            x, y = get_batch(rngs, split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
            loss, _ = loss_fn(model, x, y)
            # JAX arrays are immutable so cannot do losses[k] = loss like in pytorch.
            # The problem with this code is that losses.at[k].set(loss) creates a new
            # copy losses of len(losses). That is not memory efficient. In JAX, you
            # would want to jax.lax.scan this instead which we will cover in the full
            # jax-ified bigram.py
            losses = losses.at[k].set(loss)
        out[split] = losses.mean()
    model.train()
    return out


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, idx, targets):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    # Notice we don't require a "zero grad" operation like in pytorch where we would
    # done something like `optimizer.zero_grad()` because JAX doesn't have gradient
    # accumulation by default like pytorch. Instead, gradients are fresh per function
    # call.
    (loss, logits), grads = grad_fn(model, idx, targets)
    metrics.update(loss=loss)
    optimizer.update(model, grads)


# Super simple bigram model
model = BigramLanguageModel(rngs)
# We are using JAX, No need model.to(device)

# Train the function
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=LEARNING_RATE), wrt=nnx.Param
)
metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss")
    # accuracy=nnx.metrics.Accuracy(),
)


for iters in range(MAX_ITERS):
    model.train()

    # Every once in a while evaluate the loss on train and val sets
    if iters % EVAL_INTERVAL == 0:
        losses = estimate_loss(rngs, model)
        print(
            f"step {iters}: train loss {losses['train']:.4f}, val los {losses['val']:.4f}"
        )

    # Sample a batch of data
    xb, yb = get_batch(rngs, "train", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    # Evaluate the loss
    train_step(model, optimizer, metrics, xb, yb)

# Generate from the model
context = jnp.zeros((1, 1), dtype=jnp.int32)
print(decode(model.generate(context, max_new_tokens=500, rngs=rngs)[0].tolist()))
