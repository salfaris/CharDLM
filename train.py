import logging
import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.loader import save_checkpoint, set_ckpt_dir
from nanodlm.model import GPT, GPTConfig
from nanodlm.utils import log_model_size, log_system_info, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

ckpt_dir = set_ckpt_dir()

log_system_info()


@dataclass
class TrainConfig:
    """Configuration for training."""

    smol: bool = True

    batch_size: int = 64 if not smol else 32
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4 if not smol else 1e-3
    eval_iters: int = 200


rngs = nnx.Rngs(44)

smol = True
gpt_config = GPTConfig(smol=smol)
train_config = TrainConfig(smol=smol)

# Load dataset
dataset = load_shakespeare_dataset(train_split=0.9)
gpt_config.vocab_size = dataset.vocab_size
train_data = dataset.train_data
val_data = dataset.val_data

logger.info("--" * 12)
logger.info(f"GPT Config: {gpt_config}")
logger.info(f"Train Config: {train_config}")
logger.info("--" * 12)


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
        x, y = get_batch(
            rngs,
            data,
            train_config.eval_iters,
            train_config.batch_size,
            gpt_config.block_size,
        )

        # Scan over pre-generated indices, this pattern enables jit!
        _, losses = jax.lax.scan(eval_step, None, (x, y))  # type: ignore
        out[split] = losses.mean()

    model.train()
    return out


model = GPT(gpt_config, rngs=rngs)
log_model_size(model)

# Train the function
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=train_config.learning_rate), wrt=nnx.Param
)

metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average("loss")
    # accuracy=nnx.metrics.Accuracy(),
)

training_start_time = time.perf_counter()

for iters in range(train_config.max_iters):
    model.train()

    # Every once in a while evaluate the loss on train and val sets
    if iters % train_config.eval_interval == 0:
        losses = estimate_loss(rngs, model)

        logger.info(
            f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        # Save checkpoint
        save_checkpoint(ckpt_dir, iters, model, optimizer)

    # Sample a batch of data
    xb, yb = get_batch(
        rngs,
        train_data,
        num_samples=1,
        batch_size=train_config.batch_size,
        block_size=gpt_config.block_size,
    )
    xb = jnp.squeeze(xb, axis=0)
    yb = jnp.squeeze(yb, axis=0)

    # Evaluate the loss
    train_step(model, optimizer, metrics, xb, yb)

# Generate from the model
model.eval()

context = jnp.zeros((1, 1), dtype=jnp.int32)
logger.info("Generating from trained model:")
print("--" * 20)
print(
    model.generate_text(
        dataset,
        max_tokens=500,
        start_tokens=context[0].tolist(),
        rngs=rngs,
    )
)
print("--" * 20)

# Total time elapsed
time_elapsed = time.perf_counter() - training_start_time
logger.info(f"\nTotal training time: {time_elapsed:.2f} seconds")
