out_dir = 'out-shakespeare-gpt'
wandb_run_name = 'gpt-100B-small'
dataset = 'shakespeare'
block_size = 31 # ==> 100±3B
vocab = 'gpt2'

# model
n_layer = 6
n_head = 8
n_embd = 128