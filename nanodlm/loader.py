import orbax.checkpoint as ocp
from flax import nnx


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
