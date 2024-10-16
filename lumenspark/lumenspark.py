from transformers import PretrainedConfig, PreTrainedModel, GPT2Tokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
import torch
import math
import os

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
        seq_length=768,
        vocab_size=50257,
        embed_dim=768,
        depth=8,
        heads=12,
        dropout=1/17,
        k=384,
        rank=256,
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
        self.rank = rank

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
            "k": self.k,
            "rank": self.rank,
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
    def __init__(self, in_features, out_features, rank, init_std=0.02):
        super().__init__()
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.U.weight, std=init_std)
        nn.init.normal_(self.V.weight, std=init_std)

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
    It uses low-rank approximations to reduce computational cost and memory usage.
    """
    def __init__(self, embed_dim, num_heads, head_dim=None, dropout=0.0):
        super().__init__()
        assert (embed_dim % num_heads) == 0, 'Embedding dimension must be divisible by the number of heads'

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        # Query, Key and Value transformations using LowRankLinear
        self.q_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_heads)

        self.dropout_layer = nn.Dropout(dropout)
        self.output_transform = nn.Linear(self.head_dim * num_heads, embed_dim)

    def stable_softmax(self, x, dim=-1):
        # Subtract max for numerical stability
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / (torch.sum(exp_x, dim=dim, keepdim=True) + 1e-6)
    
    def forward(self, inputs, attention_mask=None):
        batch_size, seq_len, _ = inputs.shape

        q = self.q_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = self.stable_softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.output_transform(attention_output)

# ----------------------------
# Define Lumenspark Model Wrapper
# ----------------------------

class LumensparkModel(PreTrainedModel):
    config_class = LumensparkConfig

    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.seq_length, config.embed_dim)

        # Lumenspark transformer encoder layers with prenormalization and LayerScale
        self.layers = nn.ModuleList()
        for _ in range(config.depth):
            layer = nn.ModuleDict({
                "norm1": nn.LayerNorm(config.embed_dim),
                "attn": LumensparkSelfAttention(
                    embed_dim=config.embed_dim,
                    num_heads=config.heads,
                    head_dim=config.embed_dim // config.heads,
                    dropout=config.dropout
                ),
                "norm2": nn.LayerNorm(config.embed_dim),
                "ffn": nn.Sequential(
                    LowRankLinear(config.embed_dim, config.embed_dim * 4, rank=config.rank),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    LowRankLinear(config.embed_dim * 4, config.embed_dim, rank=config.rank),
                    nn.Dropout(config.dropout)
                ),
            })
            # Assign the parameters directly as attributes
            layer.layer_scale_attn = nn.Parameter(torch.ones(config.embed_dim) * 1e-2)
            layer.layer_scale_ffn = nn.Parameter(torch.ones(config.embed_dim) * 1e-2)
            self.layers.append(layer)

        # Final LayerNorm layer
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Feed-forward output layer
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize model weights
        self.init_weights()

    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None, **kwargs):
        """
        Downloads the pretrained weights from Hugging Face, and loads the GPT-2 tokenizer.
        """
        # Set cache directory for storing models
        cache_dir = cache_dir or os.path.join(os.getcwd(), "lumenspark_weights")

        # Download model weights in `.safetensors` format
        weight_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", cache_dir=cache_dir)

        # Load the configuration
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)
        config = LumensparkConfig.from_json_file(config_path)

        # Load GPT-2 tokenizer directly from Hugging Face
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Instantiate the model
        model = cls(config, tokenizer=tokenizer)

        # Load state_dict from safetensors file
        with safe_open(weight_path, framework="pt") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        model.load_state_dict(state_dict)

        return model

    @staticmethod
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

    def generate(self, text, max_length=160, min_length=20, temperature=0.6, top_k=50, top_p=0.9, repetition_penalty=1.1, do_sample=True):
        """
        Text generation method that handles auto-regressive generation with repetition penalty.
        The input is a string, and the output is a string generated by the model.
        """
        self.eval()  # Set model to evaluation mode
        # Tokenize input text using GPT-2 tokenizer
        input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long).to(self.device)

        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids).to(self.device)

        generated_tokens = input_ids

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(input_ids=generated_tokens, attention_mask=attention_mask)
            logits = outputs["logits"][:, -1, :]

            # Adjust temperature for randomness
            logits = logits / temperature

            # Apply repetition penalty: reduce logits of tokens that have already been generated
            for token in set(generated_tokens.view(-1).tolist()):
                logits[:, token] /= repetition_penalty  # Penalize repeated tokens

            # Apply top-k and top-p sampling to select the next token
            if do_sample:
                filtered_logits = LumensparkModel.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the generated token
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            # Update attention mask
            attention_mask = torch.ones_like(generated_tokens).to(self.device)

            # Prevent early stopping by ensuring min_length is reached before allowing EOS
            if next_token.item() == self.tokenizer.eos_token_id and generated_tokens.size(1) < min_length:
                continue  # Skip EOS if output is too short

            # Stop if the EOS token is generated and minimum length is reached
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode the generated tokens back to text
        generated_text = self.tokenizer.decode(generated_tokens[0].tolist())

        return generated_text

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

        # Create causal mask
        device = embeddings.device
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=device)).unsqueeze(0).unsqueeze(0)

        # Combine with attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match dimensions
            attention_mask = attention_mask[:, None, None, :].float()
            combined_mask = attention_mask * causal_mask
        else:
            combined_mask = causal_mask

        # Pass through each transformer layer with prenormalization and LayerScale
        for layer in self.layers:
            # Prenormalization before self-attention
            embeddings_norm = layer["norm1"](embeddings)
            attn_output = layer["attn"](embeddings_norm, attention_mask=combined_mask)
            # Apply LayerScale for attention output
            embeddings = embeddings + layer.layer_scale_attn * attn_output

            # Prenormalization before feed-forward network
            embeddings_norm = layer["norm2"](embeddings)
            ffn_output = layer["ffn"](embeddings_norm)
            # Apply LayerScale for feed-forward output
            embeddings = embeddings + layer.layer_scale_ffn * ffn_output

        # Apply final LayerNorm before output
        embeddings = self.final_norm(embeddings)

        # Compute logits (unnormalized scores)
        logits = self.fc_out(embeddings)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)

            # Base cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        return {"loss": loss, "logits": logits}
