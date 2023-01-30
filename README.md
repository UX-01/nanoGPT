# Let's build GPT: from scratch, in code, spelled out
### Notes from the Andrej Karpathy's YT video

What is the neural network under the hood that models the sequence of words in ChatGPT?

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

GPT (Generative Pretrained Tansformer) transformers do the heavy lifting under the hood.

The subset of data being used is a subset of Shakespear.

> The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of minGPT that prioritizes teeth over education. Still under active development, but currently the file train.py reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in 38 hours of training. The code itself is plain and readable: train.py is a ~300-line boilerplate training loop and model.py a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

[nanoGPT](https://github.com/karpathy/nanoGPT)

Follow the instructions in the nanoGPT repository for setup, if you're on OSX install the necessary libraries as such:

```python
# MPS acceleration is available on MacOS 12.3+
$ pip3 install torch torchvision torchaudio
$ pip3 install numpy
```

Get the tiny shakespeare dataset
```
$ curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Tokenizer Schemas


- Google created SentencePiece, which is an unsupervised text tokenizer and detokenizer mainly for Nueral Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training.
- OpenAI created tiktoken, which is a fast BPE tokeniser for use wtih OpenAI's models.






