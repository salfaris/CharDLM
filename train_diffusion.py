import logging
import time
from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.loader import save_checkpoint, set_ckpt_dir
from nanodlm.model import DLMConfig, NanoDiffusionLM
from nanodlm.utils import log_model_size, log_system_info, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

ckpt_dir = set_ckpt_dir()

log_system_info()

TrainModel = NanoDiffusionLM


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
dlm_config = DLMConfig(smol=smol)
train_config = TrainConfig(smol=smol)

# Load dataset
dataset = load_shakespeare_dataset(train_split=0.9)
dlm_config.vocab_size = dataset.vocab_size

logger.info("--" * 12)
logger.info(f"DLM Config: {dlm_config}")
logger.info(f"Train Config: {train_config}")
logger.info("--" * 12)


# We don't jit `loss_fn` because we will be using it within the training (train_step)
# and eval step (estimate_loss). When jit compiles the training/eval step, it traces
# through all operations inside the function, which in this case includes `loss_fn`.
# This reduces trace complexity.
def loss_fn(model: TrainModel, idx, targets, timesteps):
    logits = model(idx, timesteps)

    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
    return loss, logits


@nnx.jit
def train_step(
    model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, idx, targets, rngs
):
    """Train for a single step."""
    B, _ = idx.shape

    # Sample random time steps for diffusion of dim (batch_size,)
    tb = rngs.randint(shape=(B,), minval=0, maxval=dlm_config.diffusion_steps)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    # Notice we don't require a "zero grad" operation like in pytorch where we would
    # done something like `optimizer.zero_grad()` because JAX doesn't have gradient
    # accumulation by default like pytorch. Instead, gradients are fresh per function
    # call.
    (loss, logits), grads = grad_fn(model, idx, targets, tb)
    metrics.update(loss=loss, logits=logits)
    optimizer.update(model, grads)


@nnx.jit
def estimate_loss(rngs: nnx.Rngs, model):
    def eval_step(_, step_data: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        """Single eval step."""
        x, y, t = step_data
        loss, _ = loss_fn(model, x, y, t)
        return None, loss

    out = {}
    model.eval()

    for split in ["train", "val"]:
        x, y = dataset.get_batch_jit(
            rngs=rngs,
            split=cast(Literal["train", "val"], split),
            num_samples=train_config.eval_iters,
            batch_size=train_config.batch_size,
            block_size=dlm_config.block_size,
        )
        B, _, _ = x.shape
        # For jax.lax.scan scanning t together with x, y with dim (B, ...), we need t
        # to have shape (B, 1) not (B,). Otherwise, it will not scan as expected.
        t = rngs.randint(shape=(B, 1), minval=0, maxval=dlm_config.diffusion_steps)

        # Scan over pre-generated indices, this pattern enables jit!
        _, losses = jax.lax.scan(eval_step, None, (x, y, t))  # type: ignore
        out[split] = losses.mean()

    model.train()
    return out


model = NanoDiffusionLM(dlm_config, rngs=rngs)
log_model_size(model)

# Train the function
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=train_config.learning_rate), wrt=nnx.Param
)

metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

training_start_time = time.perf_counter()

for iters in range(train_config.max_iters):
    xb, yb = dataset.get_batch_jit(
        rngs=rngs,
        split="train",
        num_samples=1,
        batch_size=train_config.batch_size,
        block_size=dlm_config.block_size,
    )
    xb = jnp.squeeze(xb, axis=0)
    yb = jnp.squeeze(yb, axis=0)

    # Evaluate the loss
    train_step(model, optimizer, metrics, xb, yb, rngs=rngs)

    # Every once in a while evaluate the loss on train and val sets
    if iters % train_config.eval_interval == 0:
        losses = estimate_loss(rngs, model)

        logger.info(
            f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        # Save checkpoint
        save_checkpoint(ckpt_dir, iters, model, optimizer)

        model.train()

# Generate from the model
model.eval()

# context = jnp.zeros((1, 1), dtype=jnp.int32)
# logger.info("Generating from trained model:")
# print("--" * 20)
# print(
#     model.generate_text(
#         dataset,
#         max_tokens=500,
#         start_tokens=context[0].tolist(),
#         rngs=rngs,
#     )
# )
# print("--" * 20)

# Total time elapsed
time_elapsed = time.perf_counter() - training_start_time
logger.info(f"\nTotal training time: {time_elapsed:.2f} seconds")
