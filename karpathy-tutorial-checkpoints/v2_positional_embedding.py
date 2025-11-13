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
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
EVAL_ITERS = 300
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


class BigramLanguageModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.token_embedding_table = nnx.Embed(
            num_embeddings=VOCAB_SIZE, features=N_EMBD, rngs=rngs
        )
        self.positional_embedding_table = nnx.Embed(
            num_embeddings=BLOCK_SIZE, features=N_EMBD, rngs=rngs
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
            # Get the predictions
            logits, _ = self(idx)  # dim = (B, C)

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
