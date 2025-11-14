import logging
import time
from pathlib import Path
from typing import Literal

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.loader import load_checkpoint, save_checkpoint, set_ckpt_dir
from nanodlm.model import GPT

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s: %(message)s",  # Define the log format
    handlers=[
        logging.StreamHandler(),  # Log to the console
    ],
    force=True,
)
# Reduce Orbax logging verbosity
logging.getLogger("orbax").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)  # Orbax uses absl logging
logging.getLogger(__name__).setLevel(logging.INFO)

ckpt_dir = set_ckpt_dir()

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

# # Hyperparameters
# BATCH_SIZE = 64  # How many independent sequences will be process in parallel?
# BLOCK_SIZE = 256  # What is the maximum context length for predictions?
# MAX_ITERS = 5000
# EVAL_INTERVAL = 500
# LEARNING_RATE = 3e-4  # Decrease LR because we have deeper layers
# EVAL_ITERS = 200
# N_EMBD = 384  # Create a level of interaction
# N_HEAD = 6  # Number of attention heads
# N_LAYER = 6  # Number of block layers
# DROPOUT = 0.2
GENERATE_ONLY = False

# # Hyperparameters
BATCH_SIZE = 32  # How many independent sequences will be process in parallel?
BLOCK_SIZE = 8  # What is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
EVAL_ITERS = 200
N_EMBD = 32  # Create a level of interaction
N_HEAD = 4  # Number of attention heads
N_LAYER = 3  # Number of block layers
DROPOUT = 0.0

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

# Load dataset
dataset = load_shakespeare_dataset(train_split=0.9)
VOCAB_SIZE = dataset.vocab_size
encode = dataset.encode
decode = dataset.decode
train_data = dataset.train_data
val_data = dataset.val_data


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


# We don't jit `loss_fn` because we will be using it within the training (train_step)
# and eval step (estimate_loss). When jit compiles the training/eval step, it traces
# through all operations inside the function, which in this case includes `loss_fn`.
# This reduces trace complexity.
def loss_fn(model, idx, targets):
    logits = model(idx)

    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    loss = jnp.mean(
        # Can verify softmax using the average ~= -log(1/VOCAB_SIZE) = 4.17.
        optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    )
    return loss, logits


@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, idx, targets):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    # Notice we don't require a "zero grad" operation like in pytorch where we would
    # done something like `optimizer.zero_grad()` because JAX doesn't have gradient
    # accumulation by default like pytorch. Instead, gradients are fresh per function
    # call.
    (loss, logits), grads = grad_fn(model, idx, targets)
    metrics.update(loss=loss, logits=logits)
    optimizer.update(model, grads)


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


model = GPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout_rate=DROPOUT,
    rngs=rngs,
)

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
            save_checkpoint(ckpt_dir, iters, model, optimizer)

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
    logging.info(
        decode(model.generate(context, max_new_tokens=500, rngs=rngs)[0].tolist())
    )


else:
    logging.info("GENERATE ONLY:")
    gen_start_time = time.perf_counter()

    load_checkpoint(ckpt_dir, model, optimizer)

    # Generate from the model
    model.eval()
    logging.info("Generating from zero context (using generate_fast)")

    # Use one or few tokens from actual text, e.g. "ROMEO:"
    context_str = "ROMEO:"
    context = jnp.array([encode(context_str)], dtype=jnp.int32)
    # context = jnp.zeros((1, 1), dtype=jnp.int32)

    max_new_tokens = 500

    logging.info("FAST")
    logging.info(
        decode(
            model.generate_fast(context, max_new_tokens=max_new_tokens, rngs=rngs)[
                0
            ].tolist()
        )
    )

    time_elapsed = time.perf_counter() - gen_start_time
    logging.info(f"{time_elapsed:.2f} seconds for decode FAST.")

    logging.info("SLOW")
    logging.info(
        decode(
            model.generate(context, max_new_tokens=max_new_tokens, rngs=rngs)[
                0
            ].tolist()
        )
    )

    time_elapsed = time.perf_counter() - time_elapsed - gen_start_time
    logging.info(f"{time_elapsed:.2f} seconds for decode SLOW.")

# Total time elapsed
time_elapsed = time.perf_counter() - start_time
logging.info(f"{time_elapsed:.2f} seconds total.")
