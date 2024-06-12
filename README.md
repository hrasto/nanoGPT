This is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT).

changes: 

* reporting loss in nats per byte (instead of nats per token)
* model forward also returns attention scores of shape `(batch_size, num_layers, num_heads, seq_len, seq_len)`, but it only works when not using flash attention

To do: 

* force 2 possible vocabs: tiktoken's gpt2, or byte vocab (0-255)
* local attention
* regularized attention (entropy penalty)
