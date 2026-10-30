# Training configuration
seed = 42
batch_size = 64
block_size = 256
max_iters = 20
eval_interval = 10
learning_rate = 3e-4
device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
eval_iters = 2

# Model configuration
n_embd = 384
num_heads = 6
n_layer = 6
dropout = 0.2
