This is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT).

changes:

* reporting loss in nats per byte (instead of nats per token)
* model forward also returns attention scores of shape `(batch_size, num_layers, num_heads, seq_len, seq_len)`, but it only works when not using flash attention
* vocab/tokenization schemes
  * there are now 2 possibilities:
    1. byte (default; `ord`/`chr` => encode/decode with ascii text)
    2. gpt2 (using `tiktoken.get_encoding('gpt2')`)
  * for generation, sets automatically to byte if `config.vocab_size==256`, else sets to gpt2
  * for training, need to specify argument `vocab` (possible values `'byte'`, `'gpt2'`)

to do:

* local attention
* regularized attention (entropy penalty)
