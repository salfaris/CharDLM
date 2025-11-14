import logging
import time

import jax.numpy as jnp
from flax import nnx

from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.loader import load_model_from_checkpoint, set_ckpt_dir
from nanodlm.model import GPT, GPTConfig

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s: %(message)s",  # Define the log format
    handlers=[
        logging.StreamHandler(),  # Log to the console
    ],
    force=True,
)
# Reduce Orbax logging verbosity
logging.getLogger("orbax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)  # Orbax uses absl logging
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger(__name__).setLevel(logging.INFO)

ckpt_dir = set_ckpt_dir()

dataset = load_shakespeare_dataset()

config = GPTConfig(smol=True, vocab_size=dataset.vocab_size)
rngs = nnx.Rngs(44)

model = GPT(config, rngs=rngs)
model = load_model_from_checkpoint(ckpt_dir, model)

# Use one or few tokens from actual text, e.g. "ROMEO:"
context_str = "ROMEO: Have you"
context = jnp.array([dataset.encode(context_str)], dtype=jnp.int32)

model.eval()

generate_start_time = time.perf_counter()

logging.info("Generating from trained model:")
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
logging.info(f"{time_elapsed:.2f} seconds to generate.")
