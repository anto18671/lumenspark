from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig, GPT2Tokenizer, TrainerCallback
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
from functools import partial
from torch import nn
import random
import torch

# ----------------------------
# Custom Callback for Logging Losses
# ----------------------------

class LossLoggerCallback(TrainerCallback):
    """
    A custom callback to log the loss during training and save the plot at regular intervals,
    as well as save the final loss plot after training.
    """
    def __init__(self, plot_save_path="training_loss_plot.png", save_interval=512):
        super().__init__()
        self.losses = []
        self.plot_save_path = plot_save_path
        self.save_interval = save_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        This method is called during logging events, allowing us to extract the loss values.
        It will also save the plot every `save_interval` steps.
        """
        if 'loss' in logs:
            self.losses.append(logs['loss'])

            # Save the plot periodically
            if state.global_step % self.save_interval == 0:
                self.save_loss_plot()

    def on_train_end(self, args, state, control, **kwargs):
        """
        Save the final plot when the training ends.
        """
        print("Saving the final loss plot after training...")
        self.save_loss_plot()

    def save_loss_plot(self):
        """
        Save the plot of the logged losses to a file.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label="Training Loss")
        plt.yscale('log')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_save_path)
        plt.close()

# ----------------------------
# Define Linformer Configuration
# ----------------------------

class LinformerConfig(PretrainedConfig):
    """
    Configuration class for Linformer.
    """
    model_type = "linformer"

    def __init__(self, vocab_size, embed_dim, depth, heads, seq_length, dropout, k, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.seq_length = seq_length
        self.dropout = dropout
        self.k = k
        
# ----------------------------
# Low-Rank Linear Layer Implementation
# ----------------------------

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.V(self.U(x))
        
# ----------------------------
# Linformer Self-Attention Implementation
# ----------------------------

class LinformerSelfAttention(nn.Module):
    def __init__(self, embed_dim, max_seq_len, proj_dim, num_heads, head_dim=None, single_kv_head=True, shared_kv=True, dropout=0.1):
        super().__init__()
        assert (embed_dim % num_heads) == 0, 'Embedding dimension must be divisible by the number of heads'

        self.max_seq_len = max_seq_len
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        # Apply low-rank linear projections
        self.query_transform = nn.Sequential(
            LowRankLinear(embed_dim, embed_dim // 3, rank=48),
            nn.Linear(embed_dim // 3, self.head_dim * num_heads)
        )
        kv_size = self.head_dim if single_kv_head else (self.head_dim * num_heads)
        self.key_transform = nn.Linear(embed_dim, kv_size, bias=False)
        self.key_proj = nn.Parameter(self.initialize_proj_matrix(max_seq_len, proj_dim))

        self.shared_kv = shared_kv
        if not shared_kv:
            self.value_transform = nn.Linear(embed_dim, kv_size, bias=False)
            self.value_proj = nn.Parameter(self.initialize_proj_matrix(max_seq_len, proj_dim))

        self.dropout_layer = nn.Dropout(dropout)
        self.output_transform = nn.Linear(self.head_dim * num_heads, embed_dim)

    def initialize_proj_matrix(self, rows, cols):
        return torch.nn.init.xavier_uniform_(torch.zeros(rows, cols))

    def forward(self, inputs, context_data=None, **kwargs):
        batch_size, seq_len, _ = inputs.shape
        kv_seq_len = inputs.shape[1] if context_data is None else context_data.shape[1]
        assert kv_seq_len <= self.max_seq_len, f'Key/value sequence length exceeds the max sequence length: {self.max_seq_len}'

        # Perform linear transformation for queries
        queries = self.query_transform(inputs)

        # Perform linear transformation for keys and values
        kv_inputs = inputs if context_data is None else context_data
        keys = self.key_transform(kv_inputs)
        values = self.value_transform(kv_inputs) if not self.shared_kv else keys

        # Apply projection to reduce sequence length
        keys = torch.einsum('bnd,nk->bkd', keys, self.key_proj[:kv_seq_len])
        values = torch.einsum('bnd,nk->bkd', values, self.value_proj[:kv_seq_len] if not self.shared_kv else self.key_proj[:kv_seq_len])

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, self.proj_dim, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, self.proj_dim, -1, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (self.head_dim ** -0.5)
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Apply attention weights to values
        attention_output = torch.einsum('bhnk,bhkd->bhnd', attention_weights, values)

        # Reshape the output back to original size
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.output_transform(attention_output)
    
# ----------------------------
# RMSNorm Layer Implementation
# ----------------------------

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        """
        Root Mean Square Layer Normalization (RMSNorm) without affine transformation.
        Args:
            embed_dim: Dimensionality of the input.
            eps: Small constant to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / (x.size(-1) ** 0.5)
        return self.scale * x / (rms_x + self.eps)

# ----------------------------
# Define Linformer Model Wrapper
# ----------------------------

class LinformerModel(PreTrainedModel):
    """
    Linformer model with factorized linear projections and RMSNorm for normalization.
    """
    config_class = LinformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Token and Position Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.seq_length, config.embed_dim)

        # Linformer Transformer Encoder with LinformerSelfAttention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": LinformerSelfAttention(
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
                    LowRankLinear(config.embed_dim, config.embed_dim // 3, rank=48),
                    nn.GELU(),
                    LowRankLinear(config.embed_dim // 3, config.embed_dim, rank=48),
                    nn.Dropout(config.dropout)
                ),
                
                "norm2": RMSNorm(config.embed_dim)
            }) for _ in range(config.depth)
        ])

        # Feed-forward output layer
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

        # Dropout Layer
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
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

        # Pass through Linformer Transformer Encoder with residual connections
        for layer in self.layers:
            embeddings = layer["attn"](embeddings) + embeddings
            embeddings = layer["norm1"](embeddings)

            ffn_out = layer["ffn"](embeddings)
            embeddings = ffn_out + embeddings
            embeddings = layer["norm2"](embeddings)

        # Apply feed-forward output layer
        logits = self.fc_out(embeddings)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        return {"loss": loss, "logits": logits}

# ----------------------------
# Utility Function to Count Parameters
# ----------------------------

def count_parameters(model):
    """
    Prints the total, trainable, and non-trainable parameters of the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    
# ----------------------------
# Random Delete and Swap Operations
# ----------------------------

def random_delete(input_ids, delete_prob=0.05):
    """
    Randomly deletes tokens from the input sequence based on a specified probability.
    """
    if random.random() < delete_prob:
        delete_idx = random.randint(1, input_ids.size(1) - 2)
        input_ids = torch.cat([input_ids[:, :delete_idx], input_ids[:, delete_idx + 1:]], dim=1)
    return input_ids

def random_swap(input_ids, swap_prob=0.05):
    """
    Randomly swaps two tokens in the input sequence based on a specified probability.
    """
    if random.random() < swap_prob:
        idx1, idx2 = random.sample(range(1, input_ids.size(1) - 1), 2)
        input_ids[:, idx1], input_ids[:, idx2] = input_ids[:, idx2], input_ids[:, idx1]
    return input_ids

# ----------------------------
# Data Collator for Dynamic Chunking
# ----------------------------

def dynamic_chunking_collator(features, sequence_length, tokenizer, delete_prob=0.05, swap_prob=0.05):
    """
    Data collator that dynamically samples a chunk from each document,
    applies random delete and swap operations, and tokenizes the text.
    """
    texts = [feature['text'] for feature in features]
    
    # Tokenize the entire text with truncation and padding
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt"
    )

    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    # Apply random delete and swap operations
    input_ids = random_delete(input_ids, delete_prob=delete_prob)
    input_ids = random_swap(input_ids, swap_prob=swap_prob)

    return {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
        'attention_mask': attention_mask
    }

# ----------------------------
# Main Training Script
# ----------------------------

def main():
    # ----------------------------
    # Hyperparameters
    # ----------------------------
    SEQ_LEN = 768
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 16
    DROPOUT = 0.1
    EMBED_SIZE = 512
    NUM_HEADS = 4
    NUM_LAYERS = 6
    VOCAB_SIZE = 50257
    LEARNING_RATE = 1.25e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 5
    K = 256

    # ----------------------------
    # Device Configuration
    # ----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------------------
    # Load Dataset (OpenWebText, BookCorpus and Reddit)
    # ----------------------------
    print("Loading OpenWebText, BookCorpus and Reddit datasets...")

    # Load OpenWebText dataset
    openwebtext_dataset = load_dataset("openwebtext", split="train")

    # Load 25% of BookCorpus
    bookcorpus_dataset = load_dataset("bookcorpus", split="train[:25%]")

    # Load 10% of the Reddit dataset
    reddit_dataset = load_dataset("reddit", split="train[:10%]")

    # Combine all datasets
    combined_dataset = concatenate_datasets([openwebtext_dataset, bookcorpus_dataset, reddit_dataset])
    
    # Shuffle the combined dataset
    combined_dataset = combined_dataset.shuffle(seed=42)

    # ----------------------------
    # Load Tokenizer
    # ----------------------------
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set the padding token to the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Initialize Linformer Model
    # ----------------------------
    print("Initializing Linformer model...")
    config = LinformerConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_SIZE,
        depth=NUM_LAYERS,
        heads=NUM_HEADS,
        seq_length=SEQ_LEN,
        dropout=DROPOUT,
        k=K,
    )
    model = LinformerModel(config).to(device)

    # ----------------------------
    # Count Model Parameters
    # ----------------------------
    count_parameters(model)

    # ----------------------------
    # Define Training Arguments
    # ----------------------------
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=16,
        save_steps=2048,
        save_total_limit=8,
        warmup_steps=1024,
        remove_unused_columns=False,
        dataloader_num_workers=16,
        tf32=True,
        run_name="Linformer-OpenWebText-Training"
    )

    # ----------------------------
    # Initialize Trainer with Dynamic Chunking
    # ----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=partial(dynamic_chunking_collator, sequence_length=SEQ_LEN, tokenizer=tokenizer),
        callbacks=[LossLoggerCallback()],
    )

    # ----------------------------
    # Start Training
    # ----------------------------
    print("Starting training...")
    trainer.train()

    # ----------------------------
    # Save the Final Model
    # ----------------------------
    print("Saving the final model...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Training complete and model saved.")

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    main()
