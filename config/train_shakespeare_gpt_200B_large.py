out_dir = 'out-shakespeare-gpt'
wandb_run_name = 'gpt-200B-small'
dataset = 'shakespeare'
block_size = 62 # ==> 200±7B
vocab = 'gpt2'

# model
n_layer = 12
n_head = 16
n_embd = 256