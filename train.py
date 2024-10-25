import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import LlamaTokenizer, get_scheduler
from torch.optim import AdamW
import math
from typing import Optional, Tuple
import wandb  # Optional for logging
from tqdm import tqdm

def create_dataloader(
    dataset_name: str,
    tokenizer: LlamaTokenizer,
    batch_size: int,
    max_length: int,
    split: str = "train"
) -> DataLoader:
    """Create a dataloader from a HuggingFace dataset."""
    dataset = load_dataset(dataset_name, split=split)
    
    def tokenize_function(examples):
        # Assuming the dataset has a 'text' field - adjust if different
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
        shuffle=(split == "train")
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
        'max_length': 4096,  # Phi-3-mini context length
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
    }
    
    # Initialize wandb (optional)
    wandb.init(project="phi3-training", config=config)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Initialize your Phi-3 model (assuming you have model.py)
    from model import Phi3Model  # Your model implementation
    model = Phi3Model(
        hidden_dim=3072,
        num_heads=32,
        num_layers=32,
        vocab_size=tokenizer.vocab_size
    ).to(device)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        config['dataset_name'],
        tokenizer,
        config['batch_size'],
        config['max_length']
    )
    
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_epoch_{epoch}.pt')

if __name__ == "__main__":
    main()