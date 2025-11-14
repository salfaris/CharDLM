from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

MAX_CHECKPOINTS_TO_KEEP = 3
DEFAULT_CKPT_DIR = Path(__file__).parents[1].resolve() / "ckpt/nanodlm/"


def set_ckpt_dir(ckpt_dir: str | Path | None = None):
    """Set the global checkpoint directory."""
    if ckpt_dir is None:
        ckpt_dir = DEFAULT_CKPT_DIR
    ensure_dir(ckpt_dir)
    return ckpt_dir


def load_checkpoint(ckpt_dir, model, optimizer):
    """Load a checkpoint into model and optimizer in-place."""
    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(ckpt_dir, options=options) as read_manager:
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


def save_checkpoint(ckpt_dir, step, model, optimizer):
    """Save a checkpoint from model and optimizer."""
    ensure_dir(ckpt_dir)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=MAX_CHECKPOINTS_TO_KEEP, create=True
    )
    with ocp.CheckpointManager(ckpt_dir, options=options) as manager:
        to_save = ocp.args.Composite(
            model_state=ocp.args.PyTreeSave(item=nnx.state(model)),  # type: ignore
            optimizer_state=ocp.args.PyTreeSave(item=nnx.state(optimizer)),  # type: ignore
        )
        manager.save(step, args=to_save)


def ensure_dir(some_dir: str | Path):
    """Ensure that a directory exists."""
    Path(some_dir).mkdir(parents=True, exist_ok=True)
