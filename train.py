import os
import re
import gc
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import GPT
import hydra.utils
from tokenizer import tokenizer

torch.set_float32_matmul_precision('high')

def preprocess_corpus(corpus: str) -> str:
    return tokenizer.preprocess(corpus)

class TextDataset(Dataset):
    def __init__(self, corpus: str, context_len: int) -> None:
        super().__init__()
        self.data = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
        self.context_len = context_len
        self.size = len(self.data) - context_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        x = self.data[index : index + self.context_len]
        y = self.data[index + 1 : index + self.context_len + 1]
        return x, y

def get_dataloader(corpus: str, batch_size: int, context_len: int, cv_ratio: float, num_workers: int, device: str, shuffle: bool = True):
    dataset_obj = TextDataset(corpus, context_len)
    indices = list(range(len(dataset_obj)))
    if shuffle:
        np.random.shuffle(indices)
    split = int(len(indices) * cv_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Enable pin_memory if using CUDA
    pin_memory = True if "cuda" in device.lower() else False
    
    train_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    orig_dir = hydra.utils.get_original_cwd()
    train_file_path = os.path.join(orig_dir, cfg.train.train_file)
    
    with open(train_file_path, "r", encoding="UTF-8") as f:
        corpus = f.read()
    corpus = preprocess_corpus(corpus)
    
    train_loader, val_loader = get_dataloader(
        corpus,
        batch_size=cfg.train.batch_size,
        context_len=cfg.train.context_len,
        cv_ratio=cfg.train.cv_ratio,
        num_workers=cfg.train.num_workers,
        device=cfg.train.device
    )
    
    print("Vocab size:", tokenizer.vocab_size)
    model = GPT.create(
        tokenizer.vocab_size,
        cfg.train.context_len,
        cfg.model.n_layers,
        cfg.model.in_dim,
        cfg.model.num_heads,
        cfg.model.dropout,
        compile_model=cfg.model.compile_model
    )
    model.to(cfg.train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    print("training model on device:", cfg.train.device)
    
    for epoch in range(cfg.train.num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(cfg.train.device, non_blocking=True), y.to(cfg.train.device, non_blocking=True)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Log training loss every log_interval batches.
            if (i + 1) % cfg.train.log_interval == 0:
                avg_loss = running_loss / num_batches
                print(f"Epoch {epoch+1}, Batch {i+1}: Train Loss: {avg_loss:.4f}")
                running_loss = 0.0
                num_batches = 0
            
            del x, y, logits
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"Epoch {epoch+1} completed.")
        
        # Run validation after each epoch.
        model.eval()
        net_loss = 0.0
        cnt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.train.device, non_blocking=True), y.to(cfg.train.device, non_blocking=True)
                _, loss = model(x, y)
                net_loss += loss.item()
                cnt += 1
        avg_val_loss = net_loss / cnt if cnt > 0 else float('inf')
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        
        # Save the model checkpoint after each epoch.
        checkpoint_dir = os.path.join(orig_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()







    


