from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig, GPT2Tokenizer, TrainerCallback, GenerationConfig
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
from functools import partial
from torch import nn
import torch
import os

# ----------------------------
# Custom Callback for Logging Losses
# ----------------------------

class LossLoggerCallback(TrainerCallback):
    """
    A custom callback to log the loss during training, save the plot at regular intervals,
    and evaluate the model by generating text from predefined prompts.
    """
    def __init__(self, plot_save_path="training_loss_plot.png", plot_interval=256, eval_interval=256, prompts=None, tokenizer=None):
        super().__init__()
        self.losses = []
        self.steps = []
        self.plot_save_path = plot_save_path
        self.plot_interval = plot_interval
        self.eval_interval = eval_interval
        self.prompts = prompts if prompts is not None else []
        self.tokenizer = tokenizer

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        This method is called during logging events, allowing us to extract the loss values.
        It will also save the plot every `save_interval` steps and evaluate the model every `eval_interval` steps.
        """
        if 'loss' in logs:
            self.losses.append(logs['loss'])
            self.steps.append(state.global_step)

            # Save the plot periodically
            if state.global_step % self.plot_interval == 0:
                self.save_loss_plot()

            # Perform evaluation periodically
            if state.global_step % self.eval_interval == 0:
                print(f"\nPerforming evaluation at step {state.global_step}...")
                self.evaluate_model(kwargs['model'], self.tokenizer)

    def on_train_end(self, args, state, control, **kwargs):
        """
        Save the final plot and perform the final evaluation when the training ends.
        """
        print("\nSaving the final loss plot after training...")
        self.save_loss_plot()

        # Perform the final evaluation
        print("\nPerforming final evaluation...")
        self.evaluate_model(kwargs['model'], self.tokenizer)

    def save_loss_plot(self):
        """
        Save the plot of the logged losses to a file, plotting against actual steps.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, label="Training Loss")
        plt.yscale('log')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss (log scale)")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_save_path)
        plt.close()

    def evaluate_model(self, model, tokenizer):
        """
        Evaluate the model by generating text from predefined prompts.
        """
        model.eval()
        device = next(model.parameters()).device

        print("\nEvaluating model by generating text from prompts:")
        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    max_length=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 50)

        print("\nEvaluation complete.\n")

# ----------------------------
# Define Linformer Configuration
# ----------------------------

class LinformerConfig(PretrainedConfig):
    model_type = "linformer"

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
            LowRankLinear(embed_dim, embed_dim // 2, rank=32),
            nn.Linear(embed_dim // 2, self.head_dim * num_heads)
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
                    LowRankLinear(config.embed_dim, config.embed_dim // 2, rank=32),
                    nn.GELU(),
                    LowRankLinear(config.embed_dim // 2, config.embed_dim, rank=32),
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

        # Create GenerationConfig instance
        self.generation_config = GenerationConfig(
            max_length=128,
            min_length=16,
        )

    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        """
        Filter a distribution of logits using top-k and/or top-p filtering.
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
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
        generated_tokens = input_ids

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(input_ids=generated_tokens)
            logits = outputs["logits"][:, -1, :]
            logits = logits / temperature
            
            if do_sample:
                # Pass the top_k and top_p values only once
                filtered_logits = LinformerModel.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
        
        return generated_tokens

    def save_pretrained(self, save_directory, **kwargs):
        # Save model configuration and weights
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        
        # Save generation configuration separately
        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        config = LinformerConfig.from_pretrained(load_directory)
        model = cls(config)
        model.load_state_dict(torch.load(f"{load_directory}/pytorch_model.bin"))

        # Load generation configuration if available
        generation_config_path = f"{load_directory}/generation_config.json"
        if os.path.exists(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(load_directory)
        
        return model

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
# Data Collator for Dynamic Chunking
# ----------------------------

def dynamic_chunking_collator(features, sequence_length, tokenizer):
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
    
    SEQ_LEN = 512
    BATCH_SIZE = 48
    GRADIENT_ACCUMULATION = 10
    DROPOUT = 0.1
    EMBED_SIZE = 512
    NUM_HEADS = 4
    NUM_LAYERS = 6
    VOCAB_SIZE = 50257
    LEARNING_RATE = 1.25e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 4
    K = 128

    # ----------------------------
    # Device Configuration
    # ----------------------------
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------------------
    # Load Dataset (OpenWebText and BookCorpus)
    # ----------------------------
    
    print("Loading OpenWebText and BookCorpus datasets...")

    # Load OpenWebText dataset
    openwebtext_dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)

    # Load 30% of BookCorpus
    bookcorpus_dataset = load_dataset("bookcorpus", split="train[:30%]", trust_remote_code=True)

    # Combine all datasets
    combined_dataset = concatenate_datasets([openwebtext_dataset, bookcorpus_dataset])
    
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
    # Define Evaluation Prompts
    # ----------------------------
    prompts = [
        "Once upon a time in a land far, far away,",
        "In the future, artificial intelligence will",
        "The year is 2050, and humans have colonized Mars. The first colonists",
        "The world's largest volcano erupted, causing",
        "The secret to happiness is",
        "The president of the United States",
        "In a galaxy far, far away, a lone spaceship",
    ]

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
        warmup_steps=512,
        remove_unused_columns=False,
        dataloader_num_workers=16,
        run_name="Linformer-OpenWebText-Training",
        max_grad_norm=1.0,
        bf16=True,
    )

    # ----------------------------
    # Initialize Trainer with Dynamic Chunking
    # ----------------------------
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=partial(dynamic_chunking_collator, sequence_length=SEQ_LEN, tokenizer=tokenizer),
        callbacks=[
        LossLoggerCallback(
            prompts=prompts,
            tokenizer=tokenizer,
            plot_interval=256,
            eval_interval=256,
        )
    ],
    )

    # ----------------------------
    # Start Training
    # ----------------------------
    
    print("Starting training...")
    trainer.train()

    # ----------------------------
    # Save the Final Model and Tokenizer
    # ----------------------------
    
    print("Saving the final model...")
    trainer.save_model()
    print("Training complete and model saved.")

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    main()
