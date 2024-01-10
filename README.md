Readme

# LilGPT
LilGPT is a tiny(11.6M) generative model  that can rap. 
Trained on the lyrics of popular hip hop artists, particularly drake, this decoder based architecture model generates snippets of hip hop lyrics.

I hope to keep this repo active with contributions to optimize the currently vanilla architecture, add FlashAttention ,implement a tokenizer,(perhaps implement CUDA kernels in triton? Who knows..)

The lyrics were fetched and scraped using [*genius*](https://docs.genius.com) API, I do hope to vary the data more with future contributions.

The GPT model may generate expletives as it is a prominent part of the underlying data distribution. 
I don't really know a whole lot about letting a model generate profanity (in particular, racial slurs). If there are rulings against doing so, create an Issue and I would definitely incorporate the needful changes.

## Setup


## Installation
If you would like to use LilGPT, run:

```bash
git clone https://github.com/ammar1510/LilGPT.git
cd LilGPT
pip install -r requirements.txt
```

## Usage

To generate lyrics:

```bash
python generate.py num_tokens
```
In order to scrape songs by a particular artist, from Genius:

```bash
python scrape.py artist_name
```

If you'd like to train the model on your own dataset, load the data into `input.txt`

```bash
python train.py num_epochs
```


## References
-[MinGPT By Andrej Karpathy](https://github.com/karpathy/minGPT/tree/master)- One of the simplest implementation of a GPT model.
-[Genius API](https://docs.genius.com)- Free to use API to get songs by various artists on the open sourced platform.
-[Kaggle](https://www.kaggle.com/) provides free GPUs, which was extremely helpful in training the model.
