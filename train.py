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

def get_dataloader(corpus: str, batch_size: int, context_len: int, cv_ratio: float, num_workers: int, shuffle: bool = True):
    dataset_obj = TextDataset(corpus, context_len)
    indices = list(range(len(dataset_obj)))
    if shuffle:
        np.random.shuffle(indices)
    split = int(len(indices) * cv_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(dataset_obj, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
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
        num_workers=cfg.train.num_workers
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
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(cfg.train.device), y.to(cfg.train.device)
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del x, y, logits
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"Epoch {epoch+1}/{cfg.train.num_epochs} - Train Loss: {loss.item():.4f}")
        
        if (epoch + 1) % max(1, cfg.train.num_epochs // 10) == 0:
            model.eval()
            net_loss, cnt = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(cfg.train.device), y.to(cfg.train.device)
                    _, loss = model(x, y)
                    net_loss += loss.item()
                    cnt += 1
            print(f"Epoch {epoch+1}/{cfg.train.num_epochs} - Val Loss: {(net_loss/cnt):.4f}")

if __name__ == "__main__":
    main()







    


