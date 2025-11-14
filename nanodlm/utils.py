"""Utility functions for nanodlm."""

import logging
from typing import Any, Dict

import flax
import jax
import optax


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s: %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )

    # Reduce verbosity of third-party libraries
    logging.getLogger("orbax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger(__name__).setLevel(level)


def log_system_info() -> None:
    """Log information about JAX devices and library versions."""
    logger = logging.getLogger(__name__)

    # Log versions
    logger.info(f"flax: {flax.__version__}")
    logger.info(f"jax: {jax.__version__}")
    logger.info(f"optax: {optax.__version__}")

    # Log devices
    devices = jax.devices()
    logger.info("JAX devices found:")
    for i, device in enumerate(devices):
        logger.info(
            f"  [{i}] platform: {device.platform}, id: {device.id}, kind: {device.device_kind}"
        )

    # Log default device
    default_device = jax.devices()[0]
    logger.info(f"Default device:")
    logger.info(f"  platform: {default_device.platform}")
    logger.info(f"  id: {default_device.id}")
    logger.info(f"  kind: {default_device.device_kind}")

    # Platform-specific message
    platform_messages = {
        "gpu": "Running on GPU!",
        "tpu": "Running on TPU!",
        "cpu": "Running on CPU!",
    }
    if default_device.platform in platform_messages:
        logger.info(platform_messages[default_device.platform])


def log_model_size(model: Any) -> None:
    """Log the size of a model in parameters and memory.

    Args:
        model: A Flax NNX model
    """
    import jax
    import numpy as np
    from flax import nnx

    logger = logging.getLogger(__name__)
    params = nnx.state(model)
    total_params = sum(map(lambda x: np.prod(x.shape), jax.tree.leaves(params)))
    total_bytes_approx = total_params * 4  # assume float32, 4 bytes per param
    logger.info(
        f"Total parameters: {total_params:,} ({total_bytes_approx / 1024:.1f} KB)"
    )


def log_training_metrics(step: int, train_loss: float, val_loss: float) -> None:
    """Log training metrics in a consistent format.

    Args:
        step: Current training step
        train_loss: Training loss value
        val_loss: Validation loss value
    """
    logger = logging.getLogger(__name__)
    logger.info(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")


def log_generation(text: str, time_elapsed: float, method: str = "") -> None:
    """Log generated text with timing information.

    Args:
        text: Generated text
        time_elapsed: Time taken to generate
        method: Optional method name (e.g., "FAST", "SLOW")
    """
    logger = logging.getLogger(__name__)
    if method:
        logger.info(f"Generated text using {method}:")
    else:
        logger.info("Generated text:")
    print("--" * 20)
    print(text)
    print("--" * 20)
    logger.info(f"Generation took {time_elapsed:.2f} seconds")
