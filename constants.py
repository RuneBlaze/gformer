# Constants
MAX_TAXA = 16  # Maximum number of taxa (0-255)
VOCAB_SIZE = MAX_TAXA + 3  # 256 taxa + 2 special tokens (INTERNAL_NODE, EOS)
EMBEDDING_DIM = 768  # Following GPT2-small
NUM_HEADS = 12  # Following GPT2-small
NUM_LAYERS = 12  # Following GPT2-small
MLP_HIDDEN_DIM = 3072  # Following GPT2-small (4x embedding dim)
TREE_EMBEDDING_DIM = 768  # Same as model dimension for simplicity
MAX_SEQUENCE_LENGTH = 1024  # Following GPT2-small
MAX_GTREES = 300  # or a smaller/larger limit you know won't exceed the actual data

# Special tokens (match tokenizer values)
INTERNAL_NODE = VOCAB_SIZE - 2  # Internal node token
EOS = VOCAB_SIZE - 1  # End of sequence token
PAD = VOCAB_SIZE - 3  # Padding token
