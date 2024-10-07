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
   The Linformer architecture reduces the quadratic complexity of self-attention to linear time. This is achieved by projecting the attention matrix into a lower dimension using **low-rank projections**.  
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V$$

2. **Low-Rank Linear Projections:**  
   The model uses **LowRankLinear** layers to reduce dimensionality while maintaining model performance. This method compresses the model and reduces computational overhead in dense layers. The low-rank factorization is expressed as:  
   $$W \approx U V^\top$$  
   where \( U \in \mathbb{R}^{d \times r} \) and \( V \in \mathbb{R}^{d \times r} \), and \( r \) is the rank of the projection, \( r \ll d \).

3. **Self-Attention Mechanism in Linformer:**  
   The **LinformerSelfAttention** module reduces the complexity of self-attention by projecting keys and values into lower dimensions, significantly reducing memory and computational requirements. The attention mechanism is computed as follows:  
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V$$  
   with projected keys and values:  
   $$K' = K P, \quad V' = V P$$  
   reducing the computational complexity to \( O(n \cdot k \cdot d) \), where \( k \) is the rank of the projection.

4. **Factorized Feed-Forward Layers:**  
   Feed-forward layers are factorized to further reduce the computational burden, while still using a GELU non-linearity:  
   $$\text{FFN}(x) = W_2 \, \text{GELU}(W_1 x)$$

5. **RMSNorm for Normalization:**  
   **RMSNorm** is used for normalization, providing computational advantages over the traditional **LayerNorm** by avoiding the computation of mean and variance. The normalization is given by:  
   $$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)}$$  
   where  
   $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}$$

6. **Dropout and Residual Connections:**  
   The model incorporates dropout and residual connections around the attention and feed-forward layers to improve gradient flow and prevent overfitting.

---

### Model Components

1. **Token and Position Embeddings:**  
   Learned **token embeddings** and **positional embeddings** are summed together to produce the final input embeddings for the transformer layers:  
   $$E(x_i) = E_{\text{token}}(x_i) + E_{\text{position}}(i)$$

2. **Attention Layer:**  
   Multi-head attention is implemented with configurable heads (e.g., `NUM_HEADS = 4`), each computing attention in a reduced-dimensional subspace.

3. **Transformer Encoder Layers:**  
   Multiple Linformer layers process the input sequence, followed by normalization and feed-forward layers.

4. **Feed-Forward Output Layer:**  
   The final output of the transformer layers is passed through a linear projection layer, producing logits for token prediction.

---

### Model Hyperparameters

- **`vocab_size`**: Vocabulary size (default: 50,257).
- **`embed_dim`**: Dimensionality of embeddings (default: 512).
- **`depth`**: Number of transformer layers (default: 6).
- **`heads`**: Number of attention heads (default: 4).
- **`seq_length`**: Maximum sequence length (default: 512).
- **`dropout`**: Dropout rate (default: 0.1).
- **`k`**: Low-rank projection dimension (default: 128).

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
