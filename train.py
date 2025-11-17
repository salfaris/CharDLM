import logging
import time
from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import wandb
from chardlm.checkpoint import Checkpointer
from chardlm.dataset import load_shakespeare_dataset
from chardlm.model import CharDLM, DLMConfig
from chardlm.utils import log_model_size, log_system_info, setup_logging

# # !!! Remember to enable it again!
# jax.config.update("jax_disable_jit", True)

setup_logging()
logger = logging.getLogger(__name__)


smol = False
ckpt_name = "chardlm-smol" if smol else "chardlm-big"
checkpointer = Checkpointer(name=ckpt_name)


@dataclass
class TrainConfig:
    """Configuration for training."""

    smol: bool = smol

    batch_size: int = 128 if not smol else 32
    max_iters: int = 20000 if not smol else 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4 if not smol else 1e-3
    eval_iters: int = 200


rngs = nnx.Rngs(44)

dataset = load_shakespeare_dataset(train_split=0.9)

train_config = TrainConfig()
dlm_config = DLMConfig(
    smol=train_config.smol,
    vocab_size=dataset.vocab_size,
    mask_token_id=dataset.mask_token_id,
)

# Initialize wandb
wandb.init(
    project="chardlm",
    name=ckpt_name,
    config={
        "dlm_config": dlm_config.__dict__,
        "train_config": train_config.__dict__,
    },
)

log_system_info()
logger.info("--" * 12)
logger.info(f"DLM Config: {dlm_config}")
logger.info(f"Train Config: {train_config}")
logger.info("--" * 12)


# We don't jit `loss_fn` because we will be using it within the training (train_step)
# and eval step (estimate_loss). When jit compiles the training/eval step, it traces
# through all operations inside the function, which in this case includes `loss_fn`.
# This reduces trace complexity.
def loss_fn(
    model: CharDLM,
    idx: jax.Array,
    targets: jax.Array,
    timesteps: jax.Array,
    mask: jax.Array,
):
    logits = model(idx, timesteps)

    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    mask = mask.reshape(B * T).astype(jnp.float32)

    # Below pattern not allowed for jit
    # mask = mask.astype(jnp.bool_)
    # logits_masked = logits[mask]
    # targets_masked = targets[mask]
    # loss = jnp.mean(
    #     optax.softmax_cross_entropy_with_integer_labels(logits_masked, targets_masked)
    # )

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = per_token_loss * mask
    num_masked = mask.sum()
    loss = jnp.where(num_masked > 0, masked_loss.sum() / num_masked, 0.0)
    return loss, logits


@nnx.jit
def train_step(
    model: CharDLM,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    idx: jax.Array,
    rngs: nnx.Rngs,
):
    """Train for a single step."""
    B, _ = idx.shape

    # Sample random time steps for diffusion of dim (batch_size,)
    tb = rngs.randint(shape=(B,), minval=0, maxval=dlm_config.diffusion_steps)

    # Corrupt input
    idx_corrupted, mask = model.corrupt_input(idx, tb, rngs)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    # Notice we don't require a "zero grad" operation like in pytorch where we would
    # done something like `optimizer.zero_grad()` because JAX doesn't have gradient
    # accumulation by default like pytorch. Instead, gradients are fresh per function
    # call.
    (loss, logits), grads = grad_fn(model, idx_corrupted, idx, tb, mask)
    metrics.update(loss=loss, logits=logits)
    optimizer.update(model, grads)


@nnx.jit
def estimate_loss(rngs: nnx.Rngs, model: CharDLM):
    model.eval()

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def eval_scan(step_rng: nnx.Rngs, x: jnp.ndarray):
        """Single eval step."""
        B, _ = x.shape
        t = step_rng.randint(shape=(B,), minval=0, maxval=dlm_config.diffusion_steps)

        x_corrupted, mask = model.corrupt_input(x, t, step_rng)

        loss, _ = loss_fn(model, idx=x_corrupted, targets=x, timesteps=t, mask=mask)
        return step_rng, loss  # nnx.scan splits rngs internally

    out = {}

    for split in ["train", "val"]:
        x, _ = dataset.get_batch_jit(
            rngs=rngs,
            split=cast(Literal["train", "val"], split),
            num_samples=train_config.eval_iters,
            batch_size=train_config.batch_size,
            block_size=dlm_config.block_size,
        )

        # Scan over pre-generated indices, this pattern enables jit!
        # Scanning is done over the num_samples dimension which is the leading
        # dim by construction.
        _, losses = eval_scan(rngs, x)  # type: ignore
        out[split] = losses.mean()

    model.train()
    return out


model = CharDLM(dlm_config, rngs=rngs)
log_model_size(model)

# Train the function
optimizer = nnx.Optimizer(
    model, optax.adamw(learning_rate=train_config.learning_rate), wrt=nnx.Param
)
metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

training_start_time = time.perf_counter()

checkpointer = Checkpointer(name=ckpt_name)

for iters in range(train_config.max_iters):
    xb, _ = dataset.get_batch_jit(
        rngs=rngs,
        split="train",
        num_samples=1,
        batch_size=train_config.batch_size,
        block_size=dlm_config.block_size,
    )
    # Squeeze the num_samples dim, for train_step, num_samples is 1 anyways
    xb = jnp.squeeze(xb, axis=0)

    # Evaluate the loss
    train_step(model, optimizer, metrics, xb, rngs=rngs)

    # Every once in a while evaluate the loss on train and val sets
    if iters % train_config.eval_interval == 0:
        losses = estimate_loss(rngs, model)

        logger.info(
            f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        # Log to wandb
        wandb.log(
            {
                "train/loss": float(losses["train"]),
                "val/loss": float(losses["val"]),
                "step": iters,
            },
            step=iters,
        )

        # Save checkpoint
        checkpointer.save(iters, model, optimizer)

        model.train()

# Generate from the model
model.eval()

context_str = "ROMEO:"
context = jnp.array([dataset.encode(context_str)], dtype=jnp.int32)

logger.info("Generating from trained model:")
print("--" * 20)
print(
    dataset.decode(
        model.fast_dllm_decode(
            prompt=context[0].tolist(),
            confidence_threshold=0.9,
        ).tolist()
    )
)
print("--" * 20)

# Total time elapsed
time_elapsed = time.perf_counter() - training_start_time
logger.info(f"\nTotal training time: {time_elapsed:.2f} seconds")

# Log final metrics to wandb
wandb.log({"training/total_time_seconds": time_elapsed})
wandb.finish()
