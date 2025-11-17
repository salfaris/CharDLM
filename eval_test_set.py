"""Test script to evaluate trained model on training data samples."""

from flax import nnx

from chardlm.checkpoint import Checkpointer
from chardlm.dataset import load_shakespeare_dataset
from chardlm.model import CharDLM, DLMConfig

# Initialize RNG
rngs = nnx.Rngs(44)

# Load dataset and checkpointer
dataset = load_shakespeare_dataset()
checkpointer = Checkpointer(name="chardlm-big-256block-randomUnmaskedContextLen")
checkpointer.ckpt_dir = checkpointer.ckpt_dir.resolve()

# Create model config
config = DLMConfig(
    smol=False,
    vocab_size=dataset.vocab_size,
    mask_token_id=dataset.mask_token_id,
)
print(f"\n{'='*80}")
print(f"Configuration: {config}")
print(f"{'='*80}\n")

# Initialize and load model
model = CharDLM(config, rngs=rngs)
print("Loading model from checkpoint...")
model = checkpointer.load_model_only(model)
model.eval()
print("Model loaded and set to eval mode.\n")

# Get batch of training data
batch_size = 128
x, _ = dataset.get_batch_jit(
    rngs,
    split="val",
    num_samples=1,
    batch_size=batch_size,
    block_size=config.block_size,
)
x = x.squeeze()

# Test model on multiple samples
mask_len = 100
print(f"\n{'-'*80}")
print(f"  🧪 TESTING MODEL ON 10 SAMPLES (masking last {mask_len} tokens)")
print(f"{'-'*80}\n")

for i in range(3):
    yi = x[i]
    xi = yi[:-mask_len]

    # Main sample box header
    print(f"\n╔{'═'*78}╗")
    print(f"║ 📊 SAMPLE {i+1}/10{' '*63}║")
    print(f"╠{'═'*78}╣")

    # Context section
    context_text = dataset.decode(xi.tolist())
    context_header = f"📝 CONTEXT ({len(xi)} tokens)"
    print(f"║ {context_header}{' '*(76-len(context_header))}║")
    print(f"║{' '*78}║")
    for line in context_text.split("\n"):
        # Truncate or pad line to fit exactly 74 characters
        line = line[:74]
        print(f"║   {line:<74} ║")
    print(f"╠{'═'*78}╣")

    # Ground truth section
    ground_truth_text = dataset.decode(yi[-mask_len:].tolist())
    gt_header = f"✅ GROUND TRUTH ({mask_len} tokens)"
    print(f"║ {gt_header}{' '*(76-len(gt_header))}║")
    print(f"║{' '*78}║")
    for line in ground_truth_text.split("\n"):
        line = line[:74]
        print(f"║   {line:<74} ║")
    print(f"╠{'═'*78}╣")

    # Model prediction section
    yi_pred = model.fast_dllm_decode(
        xi.tolist(),
        confidence_threshold=0.3,
        verbose=False,
    )
    prediction_text = dataset.decode(yi_pred[-mask_len:].tolist())
    pred_header = f"🔮 MODEL PREDICTION ({mask_len} tokens)"
    print(f"║ {pred_header}{' '*(76-len(pred_header))}║")
    print(f"║{' '*78}║")
    for line in prediction_text.split("\n"):
        line = line[:74]
        print(f"║   {line:<74} ║")
    print(f"╚{'═'*78}╝")

print(f"\n{'-'*80}")
print(f"  TESTING COMPLETE!")

print(f"{'-'*80}")
