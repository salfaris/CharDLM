"""Checkpoint management for saving and loading model states."""

from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

MAX_CHECKPOINTS_TO_KEEP = 3
DEFAULT_CKPT_BASE_DIR = Path(__file__).parents[1].resolve() / "ckpt"


class Checkpointer:
    """Checkpoint manager for saving and loading model and optimizer states.

    Attributes:
        ckpt_dir: Directory where checkpoints are saved
        max_to_keep: Maximum number of checkpoints to keep (default: 3)
    """

    def __init__(
        self,
        name: str = "nanodlm",
        ckpt_dir: str | Path | None = None,
        max_to_keep: int = MAX_CHECKPOINTS_TO_KEEP,
    ):
        """Initialize the checkpointer.

        Args:
            name: Name of the checkpoint directory (used if ckpt_dir is None)
            ckpt_dir: Custom checkpoint directory path (optional)
            max_to_keep: Maximum number of checkpoints to keep
        """
        if ckpt_dir is None:
            ckpt_dir = DEFAULT_CKPT_BASE_DIR / name

        self.ckpt_dir = Path(ckpt_dir)
        self.max_to_keep = max_to_keep
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure that the checkpoint directory exists."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, model, optimizer):
        """Save a checkpoint at the given step.

        Args:
            step: Training step number
            model: Flax NNX model
            optimizer: Flax NNX optimizer
        """
        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_to_keep, create=True
        )
        with ocp.CheckpointManager(self.ckpt_dir, options=options) as manager:
            to_save = ocp.args.Composite(
                model_state=ocp.args.PyTreeSave(item=nnx.state(model)),  # type: ignore
                optimizer_state=ocp.args.PyTreeSave(item=nnx.state(optimizer)),  # type: ignore
            )
            manager.save(step, args=to_save)

    def load(self, model, optimizer):
        """Load the latest checkpoint into model and optimizer in-place.

        Args:
            model: Flax NNX model to load into
            optimizer: Flax NNX optimizer to load into

        Returns:
            Tuple of (model, optimizer) with loaded states
        """
        options = ocp.CheckpointManagerOptions(read_only=True)
        with ocp.CheckpointManager(self.ckpt_dir, options=options) as read_manager:
            step = read_manager.latest_step()
            restored = read_manager.restore(
                step,
                args=ocp.args.Composite(
                    model_state=ocp.args.PyTreeRestore(item=nnx.state(model)),  # type: ignore
                    optimizer_state=ocp.args.PyTreeRestore(item=nnx.state(optimizer)),  # type: ignore
                ),
            )

        nnx.update(model, restored["model_state"])
        nnx.update(optimizer, restored["optimizer_state"])
        return model, optimizer

    def load_model_only(self, model):
        """Load only the model state from the latest checkpoint.

        Args:
            model: Flax NNX model to load into

        Returns:
            Model with loaded state
        """
        options = ocp.CheckpointManagerOptions(read_only=True)
        with ocp.CheckpointManager(self.ckpt_dir, options=options) as read_manager:
            step = read_manager.latest_step()
            restored = read_manager.restore(
                step,
                args=ocp.args.Composite(
                    model_state=ocp.args.PyTreeRestore(item=nnx.state(model)),  # type: ignore
                ),
            )

        nnx.update(model, restored["model_state"])
        return model
