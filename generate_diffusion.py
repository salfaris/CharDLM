import logging
import time

import jax.numpy as jnp
from flax import nnx

from nanodlm.checkpoint import Checkpointer
from nanodlm.dataset import load_shakespeare_dataset
from nanodlm.model import DLMConfig, NanoDiffusionLM
from nanodlm.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

checkpointer = Checkpointer(name="nanodlm")

dataset = load_shakespeare_dataset()

dlm_config = DLMConfig(
    smol=True,
    vocab_size=dataset.vocab_size,
    mask_token_id=dataset.mask_token_id,
)
rngs = nnx.Rngs(44)

model = NanoDiffusionLM(dlm_config, rngs=rngs)
model = checkpointer.load_model_only(model)
logger.info(f"DLM Config: {dlm_config}")

# Use one or few tokens from actual text, e.g. "ROMEO:"
context_str = "ROMEO:"
context = jnp.array([dataset.encode(context_str)], dtype=jnp.int32)

model.eval()

generate_start_time = time.perf_counter()

logger.info("Generating from trained model:")
print("--" * 20)
print(
    dataset.decode(
        model.fast_dllm_decode(
            dataset,
            prompt=context[0].tolist(),
            confidence_threshold=0.9,
        ).tolist()
    )
)
print("--" * 20)

time_elapsed = time.perf_counter() - generate_start_time
logger.info(f"Generation took {time_elapsed:.2f} seconds")
