import logging
import time

import jax.numpy as jnp
from flax import nnx

from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.loader import load_model_from_checkpoint, set_ckpt_dir
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
logging.getLogger("orbax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)  # Orbax uses absl logging
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger(__name__).setLevel(logging.INFO)


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

dataset = load_shakespeare_dataset()
VOCAB_SIZE = dataset.vocab_size

rngs = nnx.Rngs(44)

ckpt_dir = set_ckpt_dir()

model = GPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout_rate=DROPOUT,
    rngs=rngs,
)
model = load_model_from_checkpoint(ckpt_dir, model)

# Generate from the model
model.eval()

# Use one or few tokens from actual text, e.g. "ROMEO:"
context_str = "ROMEO: Have you"
context = jnp.array([dataset.encode(context_str)], dtype=jnp.int32)
# context = jnp.zeros((1, 1), dtype=jnp.int32)

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
