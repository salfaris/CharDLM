import logging
import time
from pathlib import Path
from typing import Literal

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s: %(message)s",  # Define the log format
    handlers=[
        logging.StreamHandler(),  # Log to the console
    ],
    force=True,
)
# Reduce Orbax logging verbosity
logging.getLogger('orbax').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)  # Orbax uses absl logging
logging.getLogger(__name__).setLevel(logging.INFO)

ckpt_dir = Path(__file__).parent.resolve() / "ckpt/attention_ckpts/"
ckpt_dir.mkdir(exist_ok=True, parents=True)

options = ocp.CheckpointManagerOptions(
    max_to_keep=3,  # Keep only the 3 most recent checkpoints
    create=True,
)

checkpoint_manager = ocp.CheckpointManager(
    ckpt_dir,
    options=options,
)



logging.info(f"flax: {flax.__version__}")
logging.info(f"jax: {jax.__version__}")
logging.info(f"optax: {optax.__version__}")

devices = jax.devices()
logging.info("JAX devices found:")
for i, device in enumerate(devices):
    # Each device object has platform, id, and device_kind
    logging.info(
        f"  [{i}] platform: {device.platform}, id: {device.id}, kind: {device.device_kind}"
    )

# Just the first/default device you're running on:
default_device = jax.devices()[0]
logging.info(f"Default device:")
logging.info(f"  platform: {default_device.platform}")
logging.info(f"  id: {default_device.id}")
logging.info(f"  kind: {default_device.device_kind}")

# Check which platform is active
if default_device.platform == "gpu":
    logging.info("Running on GPU!")
elif default_device.platform == "tpu":
    logging.info("Running on TPU!")
elif default_device.platform == "cpu":
    logging.info("Running on CPU!")

# Hyperparameters
BATCH_SIZE = 64  # How many independent sequences will be process in parallel?
BLOCK_SIZE = 256  # What is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4  # Decrease LR because we have deeper layers
EVAL_ITERS = 200
N_EMBD = 384  # Create a level of interaction
N_HEAD = 6  # Number of attention heads
N_LAYER = 6  # Number of block layers
DROPOUT = 0.2
GENERATE_ONLY = True

# # Hyperparameters
# BATCH_SIZE = 32  # How many independent sequences will be process in parallel?
# BLOCK_SIZE = 8  # What is the maximum context length for predictions?
# MAX_ITERS = 5000
# EVAL_INTERVAL = 500
# LEARNING_RATE = 1e-3
# EVAL_ITERS = 200
# N_EMBD = 32  # Create a level of interaction
# N_HEAD = 4  # Number of attention heads
# N_LAYER = 3  # Number of block layers
# DROPOUT = 0.0

rngs = nnx.Rngs(44)

logging.info("--" * 12)
logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
logging.info(f"BLOCK_SIZE: {BLOCK_SIZE}")
logging.info(f"MAX_ITERS: {MAX_ITERS}")
logging.info(f"EVAL_INTERVAL: {EVAL_INTERVAL}")
logging.info(f"LEARNING_RATE: {LEARNING_RATE}")
logging.info(f"EVAL_ITERS: {EVAL_ITERS}")
logging.info(f"N_EMBD: {N_EMBD}")
logging.info(f"N_HEAD: {N_HEAD}")
logging.info(f"N_LAYER: {N_LAYER}")
logging.info(f"DROPOUT: {DROPOUT}")
logging.info("--" * 12)

start_time = time.perf_counter()

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


def get_batch_not_jittable(
    rngs: nnx.Rngs,
    split: Literal["train", "val"],
    block_size: int,
    batch_size: int,
):
    """Shakespeare data loader that mimics Karpathy's pytorch implementation."""
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data

    maxval = len(data) - block_size
    start_indices = rngs.randint(shape=(batch_size,), minval=0, maxval=maxval)

    x = jnp.stack([data[i : i + block_size] for i in start_indices])
    y = jnp.stack([data[i + 1 : i + 1 + block_size] for i in start_indices])

    return x, y


@nnx.jit(static_argnums=(2, 3, 4))
def get_batch(
    rngs: nnx.Rngs,
    data: jnp.ndarray,
    num_samples: int,
    batch_size: int,
    block_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shakespeare data loader that is JIT-compilable.

    Pre-generate all random indices for the batch based on num_samples, then use
    jax.vmap to extract sequences from the data array.
    """
    maxval = len(data) - block_size
    # Generate indices with dim = (num_samples, batch_size).
    all_indices = rngs.randint(shape=(num_samples, batch_size), minval=0, maxval=maxval)

    def extract_sequence(start_indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract sequences for a batch of start_indices.

        Don't let the jax.vmap and lax.dynamic_slice scare you!
        lax.dynamic_slice is just a way to do slicing with static shapes that JAX
        can reason about during JIT compilation. jax.vmap is just a way to vectorize
        operations over a batch dimension.

        This:
            x = jax.vmap(lambda idx: jax.lax.dynamic_slice(data, (idx,), (block_size,)))(
                start_indices
            )
        is equivalent to the non-jitable code below:
            x = jnp.stack([data[i : i + block_size] for i in start_indices])
        """
        x = jax.vmap(lambda idx: jax.lax.dynamic_slice(data, (idx,), (block_size,)))(
            start_indices
        )
        y = jax.vmap(
            lambda idx: jax.lax.dynamic_slice(data, (idx + 1,), (block_size,))
        )(start_indices)
        return x, y

    x, y = jax.vmap(extract_sequence)(all_indices)
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

        self.dropout = nnx.Dropout(DROPOUT, rngs=rngs)

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
        wei = self.dropout(wei)  # Randomly prevent some nodes from communicating

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
        self.proj = nnx.Linear(N_EMBD, N_EMBD, rngs=rngs)
        self.dropout = nnx.Dropout(DROPOUT, rngs=rngs)

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

    def __init__(self, n_embd: int, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(n_embd, 4 * n_embd, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                4 * n_embd, n_embd, rngs=rngs
            ),  # Projection layer into residual pathway
            nnx.Dropout(DROPOUT, rngs=rngs),
        )

    def __call__(self, x: jnp.ndarray):
        return self.net(x)


class Block(nnx.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, rngs: nnx.Rngs):
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, rngs=rngs)
        self.ffwd = FeedForward(n_embd, rngs=rngs)
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.token_embedding_table = nnx.Embed(
            num_embeddings=VOCAB_SIZE, features=N_EMBD, rngs=rngs
        )
        self.positional_embedding_table = nnx.Embed(
            num_embeddings=BLOCK_SIZE, features=N_EMBD, rngs=rngs
        )
        self.sa_heads = MultiHeadAttention(N_HEAD, N_EMBD // N_HEAD, rngs=rngs)
        self.blocks = nnx.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD, rngs=rngs) for _ in range(N_LAYER)]
        )
        self.ln_f = nnx.LayerNorm(N_EMBD, rngs=rngs)  # Final layer norm
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
            # Crop idx to the last BLOCK_SIZE tokens (since we are now doing pos encoding)
            idx_cond = idx[:, -BLOCK_SIZE:]

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
            effective_length = jnp.minimum(BLOCK_SIZE, current_pos)
            start_idx = jnp.maximum(0, current_pos - BLOCK_SIZE)
            # Crop idx to the last BLOCK_SIZE tokens (since we are now doing pos encoding)
            # idx_cond = current_idx[:, -BLOCK_SIZE:]
            idx_cond = jax.lax.dynamic_slice(tokens, (0, start_idx), (B, BLOCK_SIZE))
            # If current_pos < BLOCK_SIZE, idx_cond's right part should be padded on the left.

            # Pad idx_cond on the left if length < BLOCK_SIZE
            idx_cond = jnp.where(
                jnp.arange(BLOCK_SIZE) < BLOCK_SIZE - effective_length,
                0,
                idx_cond
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
            (output_tokens, initial_key),  # Note: idx is (B, T) array of indices in current context. 
            jnp.arange(max_new_tokens),
        )

        return final_tokens

    
# We don't jit `loss_fn` because we will be using it within the training (train_step)
# and eval step (estimate_loss). When jit compiles the training/eval step, it traces
# through all operations inside the function, which in this case includes `loss_fn`.
# This reduces trace complexity.
def loss_fn(model, idx, targets):
    logits, loss = model(idx, targets)
    return loss, logits


# We are using JAX, we don't require a context manager `torch.no_grad()` since
# JAX do not track gradients unless explicitly asked via nnx.grad or equiv.
def estimate_loss_pytorch_copy(rngs: nnx.Rngs, model):
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
def estimate_loss(rngs: nnx.Rngs, model):
    def eval_step(_, step_data: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        """Single eval step."""
        x, y = step_data
        loss, _ = loss_fn(model, x, y)
        return None, loss

    out = {}
    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        x, y = get_batch(rngs, data, EVAL_ITERS, BATCH_SIZE, BLOCK_SIZE)

        # Scan over pre-generated indices, this pattern enables jit!
        _, losses = jax.lax.scan(eval_step, None, (x, y))
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

# Compute number of parameters in model (including non-trainable weights like dropout)
params = nnx.state(model)
total_params = sum(map(lambda x: np.prod(x.shape), jax.tree.leaves(params)))
total_bytes_approx = total_params * 4  # assume float32, 4 bytes per param
logging.info(
    f"Total parameters: {total_params:,} ({total_bytes_approx / 1024:.1f}) KB\n"
)

# Train the function
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=LEARNING_RATE), wrt=nnx.Param
)

if not GENERATE_ONLY:
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss")
        # accuracy=nnx.metrics.Accuracy(),
    )


    for iters in range(MAX_ITERS):
        model.train()

        # Every once in a while evaluate the loss on train and val sets
        if iters % EVAL_INTERVAL == 0:
            losses = estimate_loss(rngs, model)

            logging.info(
                f"step {iters}: train loss {losses['train']:.4f}, val los {losses['val']:.4f}"
            )

            # Save checkpoint
            checkpoint_manager.save(
                iters,
                args=ocp.args.Composite(
                    model_state=ocp.args.PyTreeSave(nnx.state(model)),
                    optimizer_state=ocp.args.PyTreeSave(nnx.state(optimizer)),
                )
            )

        # Sample a batch of data
        xb, yb = get_batch(
            rngs,
            train_data,
            num_samples=1,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
        )
        xb = jnp.squeeze(xb, axis=0)
        yb = jnp.squeeze(yb, axis=0)

        # Evaluate the loss
        train_step(model, optimizer, metrics, xb, yb)

    # Generate from the model
    model.eval()
    context = jnp.zeros((1, 1), dtype=jnp.int32)
    logging.info(decode(model.generate(context, max_new_tokens=500, rngs=rngs)[0].tolist()))


else:
    logging.info("GENERATE ONLY:")
    gen_start_time = time.perf_counter()

    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    
    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as read_mgr:
        step = read_mgr.latest_step()
        restored = read_mgr.restore(
            step,
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(item=model_state),
                optimizer_state=ocp.args.PyTreeRestore(item=optimizer_state),
            )
        )
    
    nnx.update(model, restored["model_state"])
    nnx.update(optimizer, restored["optimizer_state"])

    # Generate from the model
    model.eval()
    logging.info("Generating from zero context (using generate_fast)")

    # Use one or few tokens from actual text, e.g. "ROMEO:"
    context_str = "ROMEO:"
    context = jnp.array([[s2i[c] for c in context_str]], dtype=jnp.int32)
    # context = jnp.zeros((1, 1), dtype=jnp.int32)

    max_new_tokens = 500

    logging.info("FAST")
    logging.info(decode(model.generate_fast(context, max_new_tokens=max_new_tokens, rngs=rngs)[0].tolist()))

    time_elapsed = time.perf_counter() - gen_start_time
    logging.info(f"{time_elapsed:.2f} seconds for decode FAST.")

    logging.info("SLOW")
    logging.info(decode(model.generate(context, max_new_tokens=max_new_tokens, rngs=rngs)[0].tolist()))

    time_elapsed = time.perf_counter() - time_elapsed - gen_start_time
    logging.info(f"{time_elapsed:.2f} seconds for decode SLOW.")

# Total time elapsed
time_elapsed = time.perf_counter() - start_time
logging.info(f"{time_elapsed:.2f} seconds total.")

checkpoint_manager.close()