from pathlib import Path

import jax.numpy as jnp


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
