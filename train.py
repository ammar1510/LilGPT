import os
import re
import gc
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from model import GPT
import hydra.utils
from tokenizer import tokenizer
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

def preprocess_corpus(corpus: str) -> str:
    return tokenizer.preprocess(corpus)

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    orig_dir = hydra.utils.get_original_cwd()
    train_file_path = os.path.join(orig_dir, cfg.train.train_file)
    
    with open(train_file_path, "r", encoding="UTF-8") as f:
        corpus = f.read()
    corpus = preprocess_corpus(corpus)
    
    logger.info(f"Vocab size: {cfg.model.vocab_size}")
    model = GPT.create(
        cfg.model.vocab_size,
        cfg.train.context_len,
        cfg.model.n_layers,
        cfg.model.in_dim,
        cfg.model.num_heads,
        cfg.model.dropout,
        compile_model=cfg.model.compile_model
    )
    model.to(cfg.train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.train.step_size, 
        gamma=cfg.train.scheduler_gamma
    )
    
    logger.info(f"Training model on device: {cfg.train.device}")

    dataset = TextDataset(corpus, cfg.train.context_len)

    val_size = int(cfg.train.cv_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = cfg.train.batch_size
    num_train_batches = cfg.train.num_train_samples 
    total_train_samples = num_train_batches * batch_size

    num_val_samples = cfg.train.num_val_samples
    total_val_samples = num_val_samples * batch_size


    if total_train_samples > len(train_dataset):
        logger.warning("Requested training samples exceed dataset size. Using replacement=True.")
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=total_train_samples
        )
    else:
        train_sampler = RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=total_train_samples
        )

    if total_val_samples > len(val_dataset):
        logger.warning("Requested validation samples exceed dataset size. Using replacement=True.")
        val_sampler = RandomSampler(
            val_dataset,
            replacement=True,
            num_samples=total_val_samples
        )
    else:
        val_sampler = RandomSampler(
            val_dataset,
            replacement=False,
            num_samples=total_val_samples
        )

    pin_mem = True if "cuda" in cfg.train.device.lower() else False


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_mem
    )

    last_completed_epoch = 0

    try:
        for epoch in range(cfg.train.num_epochs):
            model.train()
            running_loss = 0.0
            num_batches = 0
            
            for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
                x = x.to(cfg.train.device, non_blocking=True)
                y = y.to(cfg.train.device, non_blocking=True)
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % cfg.train.log_interval == 0:
                    avg_loss = running_loss / num_batches
                    logger.info(f"Epoch {epoch+1}, Batch {i+1}: Train Loss: {avg_loss:.4f}")
                    running_loss = 0.0
                    num_batches = 0

            logger.info(f"Epoch {epoch+1} training completed.")

            model.eval()
            net_loss = 0.0
            cnt = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    x = x.to(cfg.train.device, non_blocking=True)
                    y = y.to(cfg.train.device, non_blocking=True)
                    _, loss = model(x, y)
                    net_loss += loss.item()
                    cnt += 1
            avg_val_loss = net_loss / cnt if cnt > 0 else float('inf')
            logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  
            logger.info(f"Epoch {epoch+1} - Learning rate: {current_lr:.6f}")

            checkpoint_dir = os.path.join(orig_dir, cfg.generate.weights_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved model checkpoint to {checkpoint_path}")

            last_completed_epoch = epoch + 1

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught at epoch {}. Saving checkpoint before exiting.".format(last_completed_epoch))
        checkpoint_dir = os.path.join(orig_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        interrupt_checkpoint_path = os.path.join(checkpoint_dir, f"model_interrupt_epoch{last_completed_epoch}.pth")
        torch.save(model.state_dict(), interrupt_checkpoint_path)
        logger.info(f"Saved interrupt checkpoint to {interrupt_checkpoint_path}")
        return

if __name__ == "__main__":
    main()