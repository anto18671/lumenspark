from transformers import Trainer, TrainingArguments, GPT2Tokenizer, TrainerCallback
from lumenspark import LumensparkConfig, LumensparkModel
from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
from functools import partial
import torch

# ----------------------------
# Callback to Log Loss and Evaluate Model
# ----------------------------

class CustomCallback(TrainerCallback):
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
        It will also save the plot every `plot_interval` steps and evaluate the model every `eval_interval` steps.
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

        # Perform the final evaluation at the end of training
        print("\nPerforming final evaluation...")
        self.evaluate_model(kwargs['model'], self.tokenizer)

    def save_loss_plot(self):
        """
        Save the plot of the logged losses to a file, plotting loss against the training steps.
        The loss is plotted on a log scale.
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
        This method uses the tokenizer to encode the prompts, and the model to generate responses.
        """
        model.eval()
        device = next(model.parameters()).device  # Automatically detect the model's device (CPU/GPU)

        print("\nEvaluating model by generating text from prompts:")
        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():  # Disable gradient computation for evaluation
                output = model.generate(
                    input_ids, 
                    max_length=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            # Decode the generated text and display it alongside the prompt
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 50)

        print("\nEvaluation complete.\n")

# ----------------------------
# Utility Function to Count Parameters
# ----------------------------

def count_parameters(model):
    """
    Prints the total, trainable, and non-trainable parameters of the model.
    Useful for understanding model complexity.
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
    
    SEQ_LEN = 512  # Maximum sequence length for training
    BATCH_SIZE = 48  # Batch size per device
    GRADIENT_ACCUMULATION = 10  # Gradient accumulation steps
    DROPOUT = 0.1  # Dropout rate
    EMBED_SIZE = 512  # Size of the embeddings
    NUM_HEADS = 4  # Number of attention heads
    NUM_LAYERS = 6  # Number of transformer layers
    VOCAB_SIZE = 50257  # Vocabulary size (for GPT-2 tokenizer)
    LEARNING_RATE = 1.25e-4  # Learning rate
    WEIGHT_DECAY = 1e-2  # Weight decay for regularization
    EPOCHS = 4  # Number of training epochs
    K = 256  # Low-rank projection dimension for Linformer-like attention

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
    # Initialize Lumenspark Model
    # ----------------------------
    
    print("Initializing Lumenspark model...")
    config = LumensparkConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_SIZE,
        depth=NUM_LAYERS,
        heads=NUM_HEADS,
        seq_length=SEQ_LEN,
        dropout=DROPOUT,
        k=K,
    )

    # Create the Lumenspark model
    model = LumensparkModel(config).to(device)

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
        run_name="Lumenspark-OpenWebText-Training",
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
            CustomCallback(
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
    # Run the main training script
    main()
