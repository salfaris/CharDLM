# CharDLM

Character-level diffusion language model implemented in JAX.

Block decoding using NVIDIA's [Fast-dLLM](https://arxiv.org/pdf/2505.22618) algorithm (Wu et al., 2025).

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