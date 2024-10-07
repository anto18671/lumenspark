from transformers import PretrainedConfig, PreTrainedModel, GenerationConfig
from torch import nn
import torch

# ----------------------------
# Define Lumenspark Configuration
# ----------------------------

class LumensparkConfig(PretrainedConfig):
    """
    Configuration class for the Lumenspark model.
    Stores model hyperparameters like sequence length, embedding dimension, number of layers, and others.
    """
    model_type = "lumenspark"

    def __init__(
        self,
        seq_length=512,
        vocab_size=50257,
        embed_dim=512,
        depth=6,
        heads=4,
        dropout=0.1,
        k=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.seq_length = seq_length
        self.dropout = dropout
        self.k = k

    def to_dict(self):
        """
        Converts the configuration parameters to a dictionary format.
        Useful for saving the configuration or inspecting model settings.
        """
        output = super().to_dict()
        output.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "heads": self.heads,
            "seq_length": self.seq_length,
            "dropout": self.dropout,
            "k": self.k
        })
        return output

# ----------------------------
# Low-Rank Linear Layer Implementation
# ----------------------------

class LowRankLinear(nn.Module):
    """
    A low-rank linear layer that factorizes a standard linear layer into two smaller ones.
    This allows for reduced parameter count and faster computation.
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        """
        Forward pass through two low-rank linear layers (U and V).
        """
        return self.V(self.U(x))
        
# ----------------------------
# Lumenspark Self-Attention Implementation
# ----------------------------

class LumensparkSelfAttention(nn.Module):
    """
    Custom self-attention mechanism for the Lumenspark model.
    It includes low-rank approximations to reduce computational cost and memory usage.
    """
    def __init__(self, embed_dim, max_seq_len, proj_dim, num_heads, head_dim=None, single_kv_head=True, shared_kv=True, dropout=0.1):
        super().__init__()
        assert (embed_dim % num_heads) == 0, 'Embedding dimension must be divisible by the number of heads'

        self.max_seq_len = max_seq_len
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Set the dimensionality of each attention head
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        # Query transformation: Low-rank projection followed by linear layer
        self.query_transform = nn.Sequential(
            LowRankLinear(embed_dim, embed_dim // 2, rank=32),
            nn.Linear(embed_dim // 2, self.head_dim * num_heads)
        )
        kv_size = self.head_dim if single_kv_head else (self.head_dim * num_heads)
        self.key_transform = nn.Linear(embed_dim, kv_size, bias=False)
        self.key_proj = nn.Parameter(self.initialize_proj_matrix(max_seq_len, proj_dim))

        # Shared key-value projection option
        self.shared_kv = shared_kv
        if not shared_kv:
            self.value_transform = nn.Linear(embed_dim, kv_size, bias=False)
            self.value_proj = nn.Parameter(self.initialize_proj_matrix(max_seq_len, proj_dim))

        self.dropout_layer = nn.Dropout(dropout)  # Dropout for regularization
        self.output_transform = nn.Linear(self.head_dim * num_heads, embed_dim)

    def initialize_proj_matrix(self, rows, cols):
        """
        Initializes the projection matrix used to reduce the sequence length for key/value pairs.
        """
        return torch.nn.init.xavier_uniform_(torch.zeros(rows, cols))

    def forward(self, inputs, context_data=None, **kwargs):
        """
        Forward pass of the self-attention mechanism.
        """
        batch_size, seq_len, _ = inputs.shape
        kv_seq_len = inputs.shape[1] if context_data is None else context_data.shape[1]
        assert kv_seq_len <= self.max_seq_len, f'Key/value sequence length exceeds the max sequence length: {self.max_seq_len}'

        # Apply transformations to queries, keys, and values
        queries = self.query_transform(inputs)
        kv_inputs = inputs if context_data is None else context_data
        keys = self.key_transform(kv_inputs)
        values = self.value_transform(kv_inputs) if not self.shared_kv else keys

        # Apply projection matrix to keys and values
        keys = torch.einsum('bnd,nk->bkd', keys, self.key_proj[:kv_seq_len])
        values = torch.einsum('bnd,nk->bkd', values, self.value_proj[:kv_seq_len] if not self.shared_kv else self.key_proj[:kv_seq_len])

        # Reshape queries, keys, and values for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, self.proj_dim, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, self.proj_dim, -1, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attention_scores = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (self.head_dim ** -0.5)
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Apply attention weights to values and compute output
        attention_output = torch.einsum('bhnk,bhkd->bhnd', attention_weights, values)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.output_transform(attention_output)
    
# ----------------------------
# RMSNorm Layer Implementation
# ----------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) without affine transformation.
    This normalization technique scales the input without centering it.
    """
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # Small constant to prevent division by zero
        self.scale = nn.Parameter(torch.ones(embed_dim))  # Scaling factor

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        """
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / (x.size(-1) ** 0.5)  # Root mean square normalization
        return self.scale * x / (rms_x + self.eps)

# ----------------------------
# Define Lumenspark Model Wrapper
# ----------------------------

class LumensparkModel(PreTrainedModel):
    """
    Lumenspark model with factorized linear projections, multi-head attention, and RMSNorm for normalization.
    This model is specifically designed to handle long sequences efficiently.
    """
    config_class = LumensparkConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.seq_length, config.embed_dim)

        # Lumenspark transformer encoder layers
        self.layers = nn.ModuleList([nn.ModuleDict({
            "attn": LumensparkSelfAttention(
                embed_dim=config.embed_dim,
                max_seq_len=config.seq_length,
                num_heads=config.heads,
                proj_dim=config.k,
                head_dim=config.embed_dim // config.heads,
                single_kv_head=True,
                shared_kv=True,
                dropout=config.dropout
            ),
            "norm1": RMSNorm(config.embed_dim),
            "ffn": nn.Sequential(
                LowRankLinear(config.embed_dim, config.embed_dim // 2, rank=32),
                nn.GELU(),
                LowRankLinear(config.embed_dim // 2, config.embed_dim, rank=32),
                nn.Dropout(config.dropout)
            ),
            "norm2": RMSNorm(config.embed_dim)
        }) for _ in range(config.depth)])

        # Feed-forward output layer
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize model weights
        self.init_weights()

        # Create GenerationConfig instance for text generation
        self.generation_config = GenerationConfig(
            max_length=128,
            min_length=16,
        )

    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        """
        Filter a distribution of logits using top-k and/or top-p filtering.
        """
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
        return logits

    def generate(self, input_ids, max_length=128, min_length=16, temperature=1.0, top_k=50, top_p=0.95, do_sample=True):
        """
        Text generation method that handles auto-regressive generation.
        """
        generated_tokens = input_ids

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(input_ids=generated_tokens)
            logits = outputs["logits"][:, -1, :]
            logits = logits / temperature

            # Apply top-k and top-p sampling to select the next token
            if do_sample:
                filtered_logits = LumensparkModel.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the generated token
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            # Stop if the EOS token is generated
            if next_token.item() == self.config.eos_token_id:
                break

        return generated_tokens

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model. If labels are provided, the loss is also computed.
        """
        batch_size, seq_length = input_ids.size()

        # Generate position ids
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Embed tokens and positions
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # Pass through each transformer layer
        for layer in self.layers:
            embeddings = layer["attn"](embeddings) + embeddings
            embeddings = layer["norm1"](embeddings)

            ffn_out = layer["ffn"](embeddings)
            embeddings = ffn_out + embeddings
            embeddings = layer["norm2"](embeddings)

        # Compute logits (unnormalized scores)
        logits = self.fc_out(embeddings)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        return {"loss": loss, "logits": logits}