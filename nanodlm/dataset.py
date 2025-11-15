from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx


class CharacterLevelDataset:
    """Character-level dataset for language modeling.

    Attributes:
        text: Raw text data
        chars: Sorted list of unique characters in the text
        vocab_size: Number of unique characters
        s2i: Character to integer mapping
        i2s: Integer to character mapping
        train_data: Encoded training data
        val_data: Encoded validation data
    """

    def __init__(self, text: str, train_split: float = 0.9):
        """Initialize dataset from text.

        Args:
            text: Raw text data
            train_split: Fraction of data to use for training (default: 0.9)
        """
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create character mappings
        self.s2i: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.i2s: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

        # Add special mask token
        mask_token = "#"
        self.s2i[mask_token] = self.vocab_size
        self.i2s[self.vocab_size] = mask_token
        self.vocab_size += 1
        self.mask_token = mask_token
        self.mask_token_id = self.s2i[mask_token]

        # Encode and split data
        data = jnp.array(self.encode(text), dtype=jnp.int32)
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> list[int]:
        """Convert string to list of integers."""
        return [self.s2i[c] for c in s]

    def decode(self, l: list[int]) -> str:
        """Convert list of integers to string."""
        return "".join([self.i2s[i] for i in l])

    def get_batch(
        self,
        rngs: nnx.Rngs,
        split: Literal["train", "val"],
        block_size: int,
        batch_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get a batch of data (non-JIT version).

        Args:
            rngs: Random number generator
            split: Which split to use ("train" or "val")
            block_size: Sequence length
            batch_size: Batch size

        Returns:
            Tuple of (inputs, targets) each with shape (batch_size, block_size)
        """
        data = self.train_data if split == "train" else self.val_data

        maxval = len(data) - block_size
        start_indices = rngs.randint(shape=(batch_size,), minval=0, maxval=maxval)

        x = jnp.stack([data[i : i + block_size] for i in start_indices])
        y = jnp.stack([data[i + 1 : i + 1 + block_size] for i in start_indices])

        return x, y

    def get_batch_jit(
        self,
        rngs: nnx.Rngs,
        split: Literal["train", "val"],
        num_samples: int,
        batch_size: int,
        block_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get batches of data (JIT-compiled version).

        Pre-generates all random indices for multiple batches, then uses
        jax.vmap to extract sequences from the data array in a JIT-compilable way.

        Args:
            rngs: Random number generator
            split: Which split to use ("train" or "val")
            num_samples: Number of batches to generate
            batch_size: Batch size
            block_size: Sequence length

        Returns:
            Tuple of (inputs, targets) each with shape (num_samples, batch_size, block_size)
        """
        data = self.train_data if split == "train" else self.val_data
        return self._get_batch_jit_impl(rngs, data, num_samples, batch_size, block_size)

    @staticmethod
    @nnx.jit(static_argnums=(2, 3, 4))
    def _get_batch_jit_impl(
        rngs: nnx.Rngs,
        data: jnp.ndarray,
        num_samples: int,
        batch_size: int,
        block_size: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal JIT-compiled implementation of batch generation.

        This nuanced method separation is needed because nnx.jit does not support
        methods with `self` parameters. So we make this a static method that is
        called by a non-jittable get_batch_jit method that provides `self` context
        to choose between the training and validation data.
        """
        maxval = len(data) - block_size
        # Generate indices with dim = (num_samples, batch_size)
        all_indices = rngs.randint(
            shape=(num_samples, batch_size), minval=0, maxval=maxval
        )

        def extract_sequence(
            start_indices: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Extract sequences for a batch of start_indices.

            Uses lax.dynamic_slice for static shapes that JAX can reason about
            during JIT compilation. jax.vmap vectorizes over the batch dimension.

            This is equivalent to the non-jittable code:
                x = jnp.stack([data[i : i + block_size] for i in start_indices])
            """
            x = jax.vmap(
                lambda idx: jax.lax.dynamic_slice(data, (idx,), (block_size,))
            )(start_indices)
            y = jax.vmap(
                lambda idx: jax.lax.dynamic_slice(data, (idx + 1,), (block_size,))
            )(start_indices)
            return x, y

        x, y = jax.vmap(extract_sequence)(all_indices)
        return x, y

    @classmethod
    def from_file(
        cls, filepath: str | Path, train_split: float = 0.9
    ) -> "CharacterLevelDataset":
        """Load dataset from file.

        Args:
            filepath: Path to text file
            train_split: Fraction of data to use for training (default: 0.9)

        Returns:
            Dataset instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(text, train_split)


def load_shakespeare_dataset(train_split: float = 0.9) -> CharacterLevelDataset:
    """Load the Shakespeare dataset from dataset/tiny_shakespeare.txt.

    Args:
        train_split: Fraction of data to use for training (default: 0.9)

    Returns:
        Dataset instance with Shakespeare text
    """
    # Download command:
    # curl -o dataset/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    data_path = Path(__file__).parents[1] / "dataset" / "tiny_shakespeare.txt"
    return CharacterLevelDataset.from_file(data_path, train_split)
