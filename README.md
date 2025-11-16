# CharDLLM

Character-level diffusion language model implemented in JAX.

## Setup

### Dataset

Download the Tiny Shakespeare dataset:

```bash
mkdir -p dataset
curl -o dataset/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Installation

```bash
pip install -e .
```

## Usage

### Training

```bash
python train.py
```

### Generation

```bash
python generate.py
```