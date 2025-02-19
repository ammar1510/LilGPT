import sys
import torch
import hydra
from omegaconf import DictConfig
import hydra.utils
from model import GPT
from tokenizer import tokenizer  # Import tokenizer

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # The current working directory might have changed; get the original one.
    orig_dir = hydra.utils.get_original_cwd()
    device = cfg.train.device

    # Create the model using the factory method.
    model = GPT.create(
        cfg.model.vocab_size,
        cfg.train.context_len,
        cfg.model.n_layers,
        cfg.model.in_dim,
        cfg.model.num_heads,
        cfg.model.dropout,
        compile_model=cfg.model.compile_model,
    )
    model.to(device)

    # Load the model weights.
    weights_path = hydra.utils.to_absolute_path(cfg.generate.weights_path)
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights)

    # Preprocess the input text once (if needed) and then encode it.
    tokenizer.preprocess(cfg.train.train_file)  # Only if needed to build vocab beforehand.
    input_tokens = tokenizer.encode(cfg.generate.input_text)
    input_tensor = torch.tensor(input_tokens, device=device).unsqueeze(0)

    # Optionally override the number of tokens to generate via command-line.
    num_tokens = cfg.generate.num_tokens
    if len(sys.argv) > 1:
        try:
            num_tokens = int(sys.argv[1])
        except ValueError:
            pass

    generated = model.generate(input_tensor, num_tokens)
    print(tokenizer.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()



