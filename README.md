# Linformer-based Language Model

This repository contains the code and configuration to use a transformer-based language model built with a custom Linformer architecture. The model is designed to handle long-sequence tasks more efficiently by incorporating a low-rank projection mechanism for attention. This allows scaling the model to longer sequences while maintaining manageable memory and computational requirements.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture and Design](#model-architecture-and-design)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project features a Linformer-based language model designed to optimize attention mechanism efficiency, reducing the quadratic complexity typical in transformer architectures to linear complexity. The Linformer model achieves this through low-rank projections, making it ideal for processing long sequences efficiently.

The model is available for download from Hugging Face and can be easily integrated into projects via pip installation. The weights for the pre-trained model are also hosted on Hugging Face.

## Model Architecture and Design

The core of this project revolves around a **Linformer-based Transformer architecture**, which optimizes the self-attention mechanism by reducing its quadratic complexity to linear time, making it more efficient for long sequences.

### Key Design Principles

1. **Efficient Attention with Linformer:**

   - The **Linformer architecture** reduces the quadratic complexity of self-attention to linear time. In traditional transformers, the self-attention mechanism has a time complexity of $O(n^2)$, where $n$ is the sequence length. Linformer addresses this issue by projecting the attention matrix into a lower dimension using **low-rank projections**, which reduces the overall memory and computational load to $O(n)$.

   - In the standard transformer, the self-attention is computed as:
     - $Q \in \mathbb{R}^{n \times d}$ are the queries,
     - $K \in \mathbb{R}^{n \times d}$ are the keys,
     - $V \in \mathbb{R}^{n \times d}$ are the values, and
     - $d_k$ is the dimension of the keys/queries.
     - Linformer modifies this by introducing a projection matrix $P \in \mathbb{R}^{n \times k}$, reducing the dimension of $K$ and $V$: $$K' = K P, \quad V' = V P$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V$$

2. **Low-Rank Linear Projections:**

   - **LowRankLinear** is used throughout the architecture to reduce dimensionality while maintaining model expressiveness. This is achieved by factorizing the linear transformation into two smaller matrices $U$ and $V$, where: $$W \approx U V^\top$$

   - Here, $U \in \mathbb{R}^{d \times r}$ and $V \in \mathbb{R}^{d \times r}$, where $r$ is the rank of the projection. This reduces the total number of parameters in the projection from $d^2$ to $2dr$, where $r \ll d$.

   - This method helps in compressing the model, lowering the computational cost of matrix multiplications in dense layers.

3. **Self-Attention Mechanism (Modified):**

   - The **SelfAttention** module implements a multi-head self-attention mechanism without low-rank projections in this architecture. Each attention head operates on the input sequence and computes self-attention as in a standard transformer. The attention matrix remains $n \times n$, ensuring full expressivity.

   - For each attention head, the queries, keys, and values are computed as follows:

   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

   - $X \in \mathbb{R}^{n \times d}$ is the input sequence, and $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are learned projection matrices for queries, keys, and values.

   - The self-attention is then calculated using the scaled dot-product attention mechanism:

   - The complexity of this operation remains $O(n^2 \cdot d)$, as we do not reduce the attention matrix with low-rank projections.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V$$

4. **Factorized Feed-Forward Layers:**

   - Each transformer block includes a **Feed-Forward Neural Network (FFN)** that follows the attention layer. In this implementation, the FFN is factorized using **LowRankLinear** layers, reducing the computational burden of the FFN while maintaining performance.

   - The FFN consists of two linear layers with a GELU non-linearity.

   - Instead of directly projecting from $d$ to $d$, the factorized layers project from $d$ to $r$ and back to $d$, where $r$ is the reduced rank.

$$\text{FFN}(x) = W_2 \, \text{GELU}(W_1 x)$$

5. **PreNorm with LayerNorm and LayerScale:**

   - Instead of applying normalization after each module (post-norm), we use a **PreNorm** architecture where **LayerNorm** is applied before the attention and feed-forward layers. This ensures smoother gradient flow and better model stability, particularly during training.

   - In this architecture, **LayerNorm** normalizes each vector $x \in \mathbb{R}^{d}$ by subtracting the mean and dividing by the standard deviation:

   - Additionally, we incorporate **LayerScale**, a technique where a learned scaling factor is applied to the residual connection output. This helps in modulating the output of each transformer block and improves the model's ability to learn deeper representations. The output of the residual connection is scaled by a learned parameter $\lambda$:

   - The scale factor $\lambda$ is initialized to a small value (e.g., 0.1) and learned during training.

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \quad \text{where} \quad \mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}$$

$$\text{output} = \lambda \cdot \text{residual} + \text{layer}(x)$$

6. **Dropout and Residual Connections:**

   - To prevent overfitting, **dropout layers** are applied after the attention mechanism and feed-forward layers. Dropout helps regularize the model during training by randomly zeroing some of the activations.

   - **Residual connections** are included around the attention and feed-forward layers, allowing for better gradient flow during backpropagation and preventing vanishing gradients in deep networks.

---

### Model Hyperparameters

The model architecture is highly configurable through several hyperparameters:

- **`vocab_size`**: The size of the vocabulary (default: 50,257).

- **`embed_dim`**: Dimensionality of the token and positional embeddings (default: 768).

- **`depth`**: Number of Linformer transformer layers (default: 8).

- **`heads`**: Number of attention heads (default: 8).

- **`seq_length`**: Maximum sequence length (default: 768).

- **`dropout`**: Dropout rate applied throughout the network (default: 1/17).

- **`k`**: The projection dimension for the low-rank attention (default: 384).

- **`rank`**: Defines the reduced dimensionality for low-rank projections in attention, optimizing computational efficiency.

---

## Installation

To install the model, use pip:

```bash
pip install lumenspark
```

This will install the Linformer-based language model and its dependencies.

## Usage

After installing the package, you can easily load the pre-trained model and tokenizer from Hugging Face to generate text.

```python
from lumenspark import LumensparkConfig, LumensparkModel
from transformers import AutoTokenizer

# Load the configuration and model from Hugging Face
config = LumensparkConfig.from_pretrained("anto18671/lumenspark")
model = LumensparkModel.from_pretrained("anto18671/lumenspark", config=config)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("anto18671/lumenspark")

# Example input text
input_text = "Once upon a time"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
output = model.generate(
    **inputs,
    max_length=100,        # Maximum length of the generated sequence
    temperature=0.7,       # Controls randomness in predictions
    top_k=50,              # Top-k sampling to filter high-probability tokens
    top_p=0.9,             # Nucleus sampling to control diversity
    repetition_penalty=1.2 # Penalize repetition
)

# Decode and print the generated text
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

This example demonstrates loading the model and tokenizer, and generating a text sequence based on an initial prompt.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
