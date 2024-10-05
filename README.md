# Linformer-based Language Model Training from Scratch

This repository contains the code and configuration to train a transformer-based language model using a custom Linformer architecture, trained on datasets such as OpenWebText, BookCorpus, and Reddit. The model is trained from the ground up, incorporating a custom attention mechanism and dynamic chunking for optimal training performance on large-scale language datasets.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture and Design](#model-architecture-and-design)
- [Training Pipeline](#training-pipeline)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Configuration Options](#configuration-options)
- [License](#license)

## Introduction

This project demonstrates training a language model from scratch based on a Linformer architecture. The Linformer model reduces the quadratic complexity of the attention mechanism in transformers by utilizing low-rank projections, making it more efficient for long sequence processing. The training is designed to handle large datasets with dynamic chunking and a custom data collator, which ensures efficient memory use.

## Model Architecture and Design

The core of this project revolves around a **Linformer-based Transformer architecture**, which optimizes the standard self-attention mechanism found in traditional transformer models like GPT and BERT. The Linformer model introduces low-rank projections to address the inefficiencies in memory and computational overhead inherent in transformers, especially with long sequences.

---

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

3. **Self-Attention Mechanism in Linformer:**

   - The **LinformerSelfAttention** module implements a multi-head self-attention mechanism with linear projections applied to the keys and values. The attention matrix, instead of being $n \times n$, is projected into a smaller matrix of size $n \times k$, where $k$ is the rank of the projection.

   - For each attention head, the queries, keys, and values are projected into lower-dimensional spaces: $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

   - $X \in \mathbb{R}^{n \times d}$ is the input sequence, and $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are learned projection matrices.

   - The attention operation is computed as.

   - $K' = KP$ and $V' = VP$, reducing the computational complexity to $O(n \cdot k \cdot d)$.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V$$

4. **Factorized Feed-Forward Layers:**

   - Each transformer block includes a **Feed-Forward Neural Network (FFN)** that follows the attention layer. In this implementation, the FFN is factorized using **LowRankLinear** layers, reducing the computational burden of the FFN while maintaining performance.

   - The FFN consists of two linear layers with a GELU non-linearity.

   - Instead of directly projecting from $d$ to $d$, the factorized layers project from $d$ to $r$ and back to $d$, where $r$ is the reduced rank.

$$\text{FFN}(x) = W_2 \, \text{GELU}(W_1 x)$$

5. **RMSNorm for Normalization:**

   - The architecture uses **Root Mean Square Layer Normalization (RMSNorm)** instead of the traditional **LayerNorm**. RMSNorm normalizes each vector $x \in \mathbb{R}^{d}$ using the root mean square of its elements.

   - RMSNorm is computationally cheaper than LayerNorm and avoids computing mean and variance.

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \quad \text{where} \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum\_{i=1}^{d} x_i^2}$$

6. **Dropout and Residual Connections:**

   - To prevent overfitting, **dropout layers** are applied after the attention mechanism and feed-forward layers. Dropout helps regularize the model during training by randomly zeroing some of the activations.

   - **Residual connections** are included around the attention and feed-forward layers, allowing for better gradient flow during backpropagation and preventing vanishing gradients in deep networks.

---

### Model Components

1. **Token and Position Embeddings:**

   - The model uses learned **token embeddings** and **positional embeddings**. These embeddings are summed together to produce the final input embeddings that are fed into the transformer layers: $$E(x_i) = E_{\text{token}}(x_i) + E_{\text{position}}(i)$$

   - $E_{\text{token}}$ is the token embedding for token $x_i$ and $E_{\text{position}}$ is the positional embedding for the $i$-th position.

2. **Attention Layer:**

   - Each Linformer layer contains a **multi-head attention mechanism** with low-rank projections. The number of attention heads is configurable (e.g., `NUM_HEADS = 4`), where each head computes attention in a reduced dimensional subspace, and the outputs are concatenated.

3. **Transformer Encoder Layers:**

   - The model has multiple transformer layers (`NUM_LAYERS = 6` by default). Each layer applies:
     - A **LinformerSelfAttention** module that handles self-attention.
     - A **RMSNorm** for normalization.
     - A **Feed-Forward Neural Network (FFN)** to further process the attention output.

4. **Feed-Forward Output Layer:**

   - The final output of the transformer layers is passed through a linear projection layer to map it back to the vocabulary size, producing logits for token prediction: $$\text{logits} = \text{Linear}(H)$$

   - $H \in \mathbb{R}^{n \times d}$ are the hidden states from the transformer layers.

---

### Model Hyperparameters

The model architecture is highly configurable through several hyperparameters:

- **`vocab_size`**: The size of the vocabulary (default: 50,257).

- **`embed_dim`**: Dimensionality of the token and positional embeddings (default: 512).

- **`depth`**: Number of Linformer transformer layers (default: 6).

- **`heads`**: Number of attention heads (default: 4).

- **`seq_length`**: Maximum sequence length (default: 768).

- **`dropout`**: Dropout rate applied throughout the network (default: 0.1).

- **`k`**: The projection dimension for the low-rank attention (default: 256).

### Attention Complexity Breakdown

With a standard transformer, the complexity of the self-attention mechanism is $O(n^2  \cdot d)$, where:

- $n$ is the sequence length,

- $d$ is the embedding dimension.

In Linformer, the complexity reduces to $O(n \cdot k \cdot d)$, where $k$ is the projection dimension, allowing for linear scaling with sequence length.

This reduction in complexity makes Linformer suitable for longer sequences while retaining the key benefits of the transformer architecture, such as scalability, expressiveness, and the ability to handle large language models.

## Training Pipeline

The training pipeline includes:

1.  **Dataset Loading:** The model is trained on a combination of OpenWebText, BookCorpus, and Reddit datasets, dynamically chunked for efficient sequence sampling.

2.  **Loss Logging:** A custom callback logs and visualizes the training loss, periodically saving the loss curve to a file.

3.  **Custom Data Collator:** This collator applies random token deletion and swapping to add noise during training, improving the model's robustness.

4.  **Efficient Attention Mechanism:** The Linformer attention mechanism ensures that the model can scale to longer sequences while remaining computationally efficient.

## Key Components

- **Linformer Architecture:** Custom Linformer model with configurable projection dimensions, number of heads, depth, and dropout.

- **LowRankLinear:** Implements factorized linear layers for efficient projections.

- **LossLoggerCallback:** A custom Trainer callback for logging and saving loss plots during training.

- **Dynamic Chunking Collator:** A custom data collator for efficient sampling of token sequences from the dataset, along with random deletion and swapping.

- **RMSNorm:** A normalization technique that uses root mean square layer normalization instead of LayerNorm.

## Installation

To install and set up the repository, follow these steps:

1. Clone the repository:

```
git clone https://github.com/anto18671/lumenspark.git
cd lumenspark
```

2. Create a virtual environment and install dependencies:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch, you can use the `main()` function provided in the script. This function handles the entire training process, from loading datasets to saving the final trained model.

```
python train.py
```

This will start the training process using the specified datasets and Linformer model architecture. You can monitor the training process by checking the logs, which will include periodic updates about the training loss. Additionally, the loss curve will be saved as a plot file (`training_loss_plot.png`) at regular intervals.

### Saving and Loading the Model

After training, the final model and tokenizer will be saved to the `./final_model` directory. You can load the trained model and tokenizer to generate text or further fine-tune it:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the trained model and tokenizer

model = GPT2LMHeadModel.from_pretrained('./final_model')

tokenizer = GPT2Tokenizer.from_pretrained('./final_model')

# Generate text

inputs = tokenizer("Once upon a time", return_tensors="pt")

outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This snippet loads the model and tokenizer, and generates a short text continuation given an initial prompt ("Once upon a time").

## Configuration Options

The training process and model can be customized through various hyperparameters. Below are some of the key configuration options that can be modified in the `main()` function:

- **`SEQ_LEN`**: Controls the maximum sequence length for each input. Longer sequences can capture more context but require more memory.

```python
SEQ_LEN = 768
```

- **`BATCH_SIZE`**: Determines how many samples are processed at once in each training step.

```python
BATCH_SIZE = 32
```

- **`EMBED_SIZE`**: Dimensionality of the token and positional embeddings. A higher embed size can improve expressiveness but increases the model size.

```python
EMBED_SIZE = 512
```

- **`NUM_HEADS`**: Number of attention heads in the multi-head attention mechanism.

```python
NUM_HEADS = 4
```

- **`NUM_LAYERS`**: Number of transformer layers in the model. More layers can increase the depth and complexity of the model but require more computational resources.

```python
NUM_LAYERS = 6
```

- **`DROPOUT`**: Dropout rate used to prevent overfitting during training. It helps regularize the model by randomly dropping neurons during training.

```python
DROPOUT = 0.1
```

- **`LEARNING_RATE`**: The initial learning rate for the optimizer. A smaller learning rate can lead to more stable training but slower convergence.

```python
LEARNING_RATE = 1.25e-4
```

- **`WEIGHT_DECAY`**: Weight decay is used to prevent overfitting by penalizing large weights.

```python
WEIGHT_DECAY = 1e-2
```

You can modify these hyperparameters directly in the `main()` function or pass them as command-line arguments when launching the training script.

## Training Performance

The training process is optimized for handling large datasets by:

1.  **Gradient Accumulation:** Accumulates gradients over multiple batches before performing an update, effectively allowing training with a larger batch size while saving GPU memory.

2.  **Dynamic Chunking:** The custom collator dynamically samples a chunk from each document, ensuring that the input sequence fits within the specified length (`SEQ_LEN`) and avoiding memory overflow.

3.  **Low-Rank Projections:** Reduces the computational cost of the attention mechanism through the use of low-rank approximations, making the model more efficient during training, especially with long sequences.

4.  **Custom Loss Logging Callback:** A callback function logs the training loss at regular intervals and periodically saves a plot of the loss, helping you monitor the training process visually.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
