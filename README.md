This is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT).

changes: 

* reporting loss in nats per byte (instead of nats per token)

To do: 

* force 2 possible vocabs: tiktoken's gpt2, or byte vocab (0-255)
* local attention
* regularized attention (entropy penalty)
