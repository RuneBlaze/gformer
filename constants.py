# Constants
MAX_TAXA = 16  # Maximum number of taxa (0-255)
VOCAB_SIZE = 260  # 256 taxa + 4 special tokens (matches tokenizer)
EMBEDDING_DIM = 768  # Following GPT2-small
NUM_HEADS = 12  # Following GPT2-small
NUM_LAYERS = 12  # Following GPT2-small
MLP_HIDDEN_DIM = 3072  # Following GPT2-small (4x embedding dim)
TREE_EMBEDDING_DIM = 768  # Same as model dimension for simplicity
MAX_SEQUENCE_LENGTH = 1024  # Following GPT2-small

# Special tokens (match tokenizer values)
LEFT_PAREN = 256  # ( token
RIGHT_PAREN = 257  # ) token
END_OF_INPUT = 258  # EOI token
END_OF_OUTPUT = 259  # EOS token