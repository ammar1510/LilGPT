import re
from typing import List, Callable

class Tokenizer:
    """
    A simple character-level tokenizer that preprocesses text,
    builds a vocabulary, and provides encoding/decoding functions.
    """

    def __init__(self) -> None:
        self.encoder: Callable[[str], List[int]] = lambda s: []
        self.decoder: Callable[[List[int]], str] = lambda tokens: ""
        self.vocab_size: int = 0
        self.vocab: List[str] = []
        self.char_to_index = {}
        self.index_to_char = {}

    def preprocess(self, corpus: str) -> str:
        """
        Cleans the input corpus, builds a vocabulary, and sets up the
        encoder and decoder functions. Returns the cleaned text.
        """
        cleaned_text = re.sub(
            r'[^a-zA-Z0-9&.[\]()!{}:"\'/\\,]', ' ', corpus
        )
        self.vocab = sorted(set(cleaned_text))
        self.vocab_size = len(self.vocab)
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.encoder = lambda s: [self.char_to_index[w] for w in s]
        self.decoder = lambda tokens: ''.join([self.index_to_char[t] for t in tokens])
        return cleaned_text

    def encode(self, s: str) -> List[int]:
        """Encodes a string into a list of integers (tokens)."""
        return self.encoder(s)

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of integers (tokens) back into a string."""
        return self.decoder(tokens)
    
    def get_vocab_size(self, corpus: str) -> int:
        """
        Given a corpus, preprocess it to build the vocabulary if necessary,
        and return the number of unique tokens (vocabulary size).
        """
        self.preprocess(corpus)
        return self.vocab_size

# Create a global tokenizer instance that you can import easily across scripts.
tokenizer = Tokenizer() 

if __name__ == "__main__":
    data_path = "data/lyrics.txt"
    with open(data_path, "r", encoding="UTF-8") as f:
        corpus = f.read()
    vocab_size = tokenizer.get_vocab_size(corpus)
    print(f"Size of vocabulary for {data_path}: {vocab_size}")
