# Lumenspark: Low-Rank Language Model

This repository contains the code and configuration to use a transformer-based language model built with low-rank projections and attention optimizations. The Lumenspark model introduces novel techniques to efficiently handle long-sequence tasks, reducing computational complexity without sacrificing model capacity.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture and Design](#model-architecture-and-design)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Lumenspark is a transformer model designed for long-sequence tasks, leveraging low-rank approximations in its linear transformations and attention mechanisms. By applying low-rank factorization, we reduce the number of parameters and the overall computational cost of the model, making it ideal for tasks that require efficient processing of large-scale data. Lumenspark is built to be scalable while retaining strong performance for natural language generation and other transformer-based tasks.

## Model Architecture and Design

Lumenspark uses a custom transformer architecture with low-rank approximations. It aims to reduce memory usage and improve inference speed without compromising on model expressiveness.

### Key Design Principles

1. **Low-Rank Linear Projections:**

   - Lumenspark uses **LowRankLinear** layers throughout the architecture to reduce the dimensionality of matrix multiplications.
   - A typical matrix multiplication $W \in \mathbb{R}^{d 	imes d}$ is approximated using two smaller matrices $U \in \mathbb{R}^{d 	imes r}$ and $V \in \mathbb{R}^{d 	imes r}$ where $r$ is the rank of the projection. This reduces the parameter count significantly:
     $$W pprox U V^	op$$
     $$
     ext{Parameter Count Reduction: } d^2
     ightarrow 2dr
     $$

2. **Self-Attention with Low-Rank Projections:**

   - **LumensparkSelfAttention** module implements a multi-head self-attention mechanism where low-rank approximations are applied to both the key and value projections, reducing the quadratic complexity of the attention mechanism to linear.
   - In standard attention:
     $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
     $$
     ext{Attention}(Q, K, V) = ext{softmax}\left(rac{Q K^ op}{\sqrt{d_k}}
     ight)V
     $$
   - With Lumenspark's low-rank modification, the keys and values are projected:
     $$K' = K P, \quad V' = V P$$
     $$ ext{where } P \in \mathbb{R}^{n imes k}$$

   - This reduces computational cost to $O(nkd)$.

3. **Prenormalization and LayerScale:**

   - Lumenspark employs **prenormalization** and **LayerScale** techniques in its transformer layers. Prenormalization applies layer normalization before attention and feed-forward networks, ensuring stable gradients during training.
   - LayerScale introduces learnable scaling parameters for both the attention and feed-forward layers:
     $$H = H + 	ext{LayerScale}_	ext{attn} \cdot 	ext{Attn}(H)$$
     $$H = H + 	ext{LayerScale}_	ext{ffn} \cdot 	ext{FFN}(H)$$

4. **Factorized Feed-Forward Layers:**

   - The feed-forward layers in each transformer block are also factorized using **LowRankLinear** to further reduce the number of parameters:
     $$ ext{FFN}(H) = W_2 \, ext{GELU}(W_1 H)$$
     - where $W_1$ and $W_2$ are low-rank projections.

5. **Dropout and Residual Connections:**

   - **Dropout** is applied after each attention and feed-forward layer to prevent overfitting, and **residual connections** are used throughout the model to improve gradient flow.

---

### Model Components

1. **Token and Positional Embeddings:**

   - Lumenspark uses learned **token embeddings** and **positional embeddings**, which are summed together to create input embeddings for the transformer layers:
     $$E(x_i) = E_{	ext{token}}(x_i) + E_{	ext{position}}(i)$$

2. **Self-Attention Module:**

   - Each transformer block includes a multi-head self-attention module with low-rank projections applied to the keys and values. This reduces memory consumption while maintaining attention effectiveness.

3. **Feed-Forward Network (FFN):**

   - Each block contains a factorized FFN, which applies GELU activation between low-rank linear projections.

4. **Layer Normalization and LayerScale:**

   - Layer normalization is applied before each self-attention and FFN, and LayerScale parameters are used to scale the outputs of each submodule.

---

## Installation

To install the Lumenspark package, use pip:

```bash
pip install lumenspark
```

## Usage

After installing, you can load the model and tokenizer from Hugging Face Hub to generate text. Here’s an example:

```python
from lumenspark import LumensparkModel, LumensparkConfig
from transformers import GPT2Tokenizer

# Load the Lumenspark model and configuration
config = LumensparkConfig.from_pretrained("lumenspark-model-id")
model = LumensparkModel.from_pretrained("lumenspark-model-id", config=config)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input text
input_text = "The future of AI is"

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
generated_text = model.generate(
    text=input_text,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
)

# Print the generated text
print(generated_text)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
