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
   - A typical matrix multiplication is approximated as follows:

     - $W \in \mathbb{R}^{d 	imes d}$ is approximated by two smaller matrices:
     - $U \in \mathbb{R}^{d 	imes r}$ and $V \in \mathbb{R}^{d 	imes r}$
     - where $r$ is the rank of the projection.
     - This reduces the parameter count from $d^2$ to $2dr$, allowing for more efficient computation.

     $$W pprox U V^	op$$

2. **Self-Attention with Low-Rank Projections:**

   - In the standard transformer, the self-attention is computed as:
     - $Q \in \mathbb{R}^{n 	imes d}$ are the queries,
     - $K \in \mathbb{R}^{n 	imes d}$ are the keys,
     - $V \in \mathbb{R}^{n 	imes d}$ are the values, and
     - $d_k$ is the dimension of the keys/queries.
   - Lumenspark modifies this by introducing a projection matrix:

     - $P \in \mathbb{R}^{n 	imes k}$ to reduce the dimensionality of $K$ and $V$:

     $$K' = K P, \quad V' = V P$$

   - This reduces the computational cost of attention from quadratic $O(n^2d)$ to linear $O(nkd)$.

   The attention mechanism is then computed as:

   $$
   	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{Q K^	op}{\sqrt{d_k}}
   ight)V
   $$

3. **Prenormalization and LayerScale:**

   - Lumenspark employs **prenormalization** and **LayerScale** in its transformer layers:

     - **Prenormalization** applies layer normalization before the attention and feed-forward network (FFN) to stabilize training.
     - **LayerScale** adds learnable scaling parameters to the output of each attention and FFN module:

     $$H = H + 	ext{LayerScale}_	ext{attn} \cdot 	ext{Attention}(H)$$

     $$H = H + 	ext{LayerScale}_	ext{ffn} \cdot 	ext{FFN}(H)$$

4. **Factorized Feed-Forward Layers:**

   - In each transformer block, the **Feed-Forward Network (FFN)** is factorized using **LowRankLinear** projections:

     - The FFN projects the input $H$ into a higher dimension using $W_1$, applies a **GELU** activation, and projects it back to the original dimension using $W_2$:

     $$ ext{FFN}(H) = W_2 \, ext{GELU}(W_1 H)$$

     - Both $W_1$ and $W_2$ are factorized to reduce the number of parameters.

5. **Dropout and Residual Connections:**

   - **Dropout** is applied after the attention and FFN layers to prevent overfitting.
   - **Residual connections** are used throughout to ensure gradient flow and prevent vanishing gradients.

---

### Model Components

1. **Token and Positional Embeddings:**

   - Lumenspark uses learned **token embeddings** and **positional embeddings**.
   - These embeddings are combined and fed into the transformer layers:

     $$E(x_i) = E_{	ext{token}}(x_i) + E_{	ext{position}}(i)$$

2. **Self-Attention Module:**

   - Each transformer block includes a multi-head self-attention module with low-rank projections applied to the keys and values, reducing memory consumption while maintaining performance.

3. **Feed-Forward Network (FFN):**

   - The **FFN** within each block applies GELU activation between low-rank projections, with factorized layers to further reduce parameter count.

4. **Layer Normalization and LayerScale:**

   - **Layer normalization** is applied before each self-attention and FFN module.
   - **LayerScale** parameters are applied to the outputs of attention and FFN modules to modulate the layer outputs.

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
