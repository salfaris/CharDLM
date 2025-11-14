import logging
import time

import jax.numpy as jnp
from flax import nnx

from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.loader import load_model_from_checkpoint, set_ckpt_dir
from nanodlm.model import GPT, GPTConfig
from nanodlm.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ckpt_dir = set_ckpt_dir()

dataset = load_shakespeare_dataset()

gpt_config = GPTConfig(smol=True, vocab_size=dataset.vocab_size)
rngs = nnx.Rngs(44)

model = GPT(gpt_config, rngs=rngs)
model = load_model_from_checkpoint(ckpt_dir, model)
logger.info(f"GPT Config: {gpt_config}")

# Use one or few tokens from actual text, e.g. "ROMEO:"
context_str = "ROMEO: Have you"
context = jnp.array([dataset.encode(context_str)], dtype=jnp.int32)

model.eval()

generate_start_time = time.perf_counter()

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

time_elapsed = time.perf_counter() - generate_start_time
logger.info(f"Generation took {time_elapsed:.2f} seconds")
