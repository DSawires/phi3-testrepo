import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
import math
from typing import Optional, Tuple
import wandb
from tqdm import tqdm

def create_dataloader(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    split: str = "train"
) -> DataLoader:
    """Create a dataloader from a HuggingFace dataset."""
    dataset = load_dataset(dataset_name, split=split)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4
    )

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Prepare labels for causal language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padded tokens
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log to wandb if enabled
        if wandb.run is not None:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    return total_loss / len(dataloader)

def main():
    # Configuration
    config = {
        'dataset_name': 'roneneldan/TinyStories',  # Replace with your dataset
        'max_length': 4096,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
    }
    
    # Initialize wandb (optional)
    wandb.init(project="phi3-training", config=config)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")
    
    # Initialize your Phi-3 model
    print("Initializing model...")
    from model import Transformer as Phi3Model  # Your model implementation
    model = Phi3Model(
        vocab_size=tokenizer.vocab_size
    ).to(device)
    print("Model initialized")
    
    # Create dataloaders
    print("Creating dataloader...")
    train_dataloader = create_dataloader(
        config['dataset_name'],
        tokenizer,
        config['batch_size'],
        config['max_length']
    )
    print(f"Dataloader created with {len(train_dataloader)} batches")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize learning rate scheduler
    num_training_steps = len(train_dataloader) * config['num_epochs']
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        avg_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            device,
            epoch
        )
        
        print(f"Epoch {epoch} average loss: {avg_loss}")
        
        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()